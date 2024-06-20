import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch.distributions.normal import Normal

import signatory

from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import MessagePassing


# region PointNet++
class PointNetLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        # Message passing with "max" aggregation.
        super().__init__(aggr="max")

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden
        # node dimensionality plus point dimensionality (=3).
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),  # szeng needs this as arg
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(
        self,
        h_j: torch.Tensor,
        pos_j: torch.Tensor,
        pos_i: torch.Tensor,
    ) -> torch.Tensor:
        # h_j: The features of neighbors as shape [num_edges, in_channels]
        # pos_j: The position of neighbors as shape [num_edges, 3]
        # pos_i: The central node position as shape [num_edges, 3]

        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointNet(torch.nn.Module):
    def __init__(self, h_dim: int = 32):
        super().__init__()

        self.conv1 = PointNetLayer(3, h_dim)
        self.conv2 = PointNetLayer(h_dim, h_dim)

    def forward(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        # Perform two-layers of message passing:
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]
        return h
# endregion

class SignatureHead(nn.Module):
    def __init__(self, in_channels: int, sig_depth: int):
        super(SignatureHead, self).__init__()
        self.augment = signatory.Augment(
            in_channels=in_channels,
            layer_sizes=(8, 8, 2),
            kernel_size=4,
            include_original=True,
            include_time=True,
        )
        self.signature = signatory.Signature(depth=sig_depth)
        # +3 because signatory.Augment is used to add time, and 2 other channels,
        # as well
        self.sig_channels = signatory.signature_channels(
            channels=in_channels + 3, depth=sig_depth
        )

    def get_outdim(self):
        return self.sig_channels

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.augment(inp)
        if x.size(1) <= 1:
            raise RuntimeError(
                "Given an input with too short a stream to take the signature"
            )
        y = self.signature(x, basepoint=True)
        return y


class LatentStateHead(nn.Module):
    def __init__(self, in_channels: int, type:str="last"):
        super(LatentStateHead, self).__init__()
        assert type in ["last", "mean"]
        self.type = type
        self.outdim = in_channels

    def get_outdim(self) -> int:
        return self.outdim

    def forward(self, x):
        if self.type == "last":
            return x[:, -1, :]
        elif self.type == "mean":
            return x.mean(dim=1)
        else:
            raise NotImplementedError()


class TDABackbone(nn.Module):
    """Runs PH vectorizations through an mTAN encoder."""
    def __init__(self, args):
        super(TDABackbone, self).__init__()
        self.num_timepts = args.num_timepts
        self.recog_net = VecRecogNet(
            mtan_input_dim=args.vec_inp_dim,
            mtan_hidden_dim=args.mtan_h_dim,
            latent_dim=args.z_dim,
            use_atanh=False,
        )

    def forward(self, batch, device):
        parts = {key: val.to(device) for key, val in batch.items()}
        parts_inp_obs = parts["inp_obs"]
        parts_inp_msk = parts["inp_msk"]
        parts_inp_tps = parts["inp_tps"]
        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)
        return self.recog_net(inp), parts["evd_obs"], parts["evd_msk"], parts["aux_obs"]


class TDABare(nn.Module):
    """Directly passes PH vectorizations (without any recognition network) forward."""
    def __init__(self, args):
        super(TDABare, self).__init__()
        self.num_timepts = args.num_timepts

    def forward(self, batch, device):
        parts = {key: val.to(device) for key, val in batch.items()}
        parts_inp_obs = parts["inp_obs"]
        parts_inp_msk = parts["inp_msk"]
        parts_inp_tps = parts["inp_tps"]
        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)

        return inp, parts["evd_obs"], parts["evd_msk"], parts["aux_obs"]


class PointNetBackbone(nn.Module):
    """Runs point clouds through a PointNet++ and then through an mTAN encoder."""
    def __init__(self, args):
        super(PointNetBackbone, self).__init__()
        self.num_timepts = args.num_timepts
        self.point_net = PointNet(h_dim=args.pointnet_dim)
        self.recog_net = VecRecogNet(
            mtan_input_dim=args.pointnet_dim,
            mtan_hidden_dim=args.mtan_h_dim,
            latent_dim=args.z_dim,
            use_atanh=False,
        )

    def forward(self, batch, device):
        pts_msk_batch = batch["pts_msk_batch"].to(device)
        pts_tid_batch = batch["pts_tid_batch"].to(device)
        pts_aux_batch = batch["pts_aux_batch"].to(device)
        pts_cut_batch = batch["pts_cut_batch"]

        pts_obs_batch = batch["pts_obs_batch"]
        pts_obs_batch = Batch.from_data_list(pts_obs_batch)
        pts_obs_batch = pts_obs_batch.to(device)

        enc = self.point_net(
            pts_obs_batch.pos, pts_obs_batch.edge_index, pts_obs_batch.batch
        )
        enc = enc.tensor_split(pts_cut_batch, dim=0)
        enc = torch.stack(enc)

        N, T, D = enc.shape
        parts_inp_obs = torch.zeros(N, self.num_timepts, D, device=device)
        parts_inp_msk = torch.zeros(N, self.num_timepts, D, device=device)
        parts_inp_tps = torch.zeros(N, self.num_timepts, device=device)
        parts_inp_obs[:, :T] = enc
        parts_inp_tps[:, :T] = pts_tid_batch / self.num_timepts
        parts_inp_msk[:, :T] = 1
        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)

        pts_tid_batch = pts_tid_batch.view(
            pts_tid_batch.shape + torch.Size([1])
        ).expand_as(enc)
        evd_obs = torch.zeros(N, self.num_timepts, D, device=device)
        evd_obs.scatter_(1, pts_tid_batch, enc)
        evd_msk = pts_msk_batch.expand(N, self.num_timepts, D)

        return self.recog_net(inp), evd_obs, evd_msk, pts_aux_batch


class PointNetBare(nn.Module):
    """Runs point clouds through a PointNet++ and then directly passes the output forward."""
    def __init__(self, args):
        super(PointNetBare, self).__init__()
        self.num_timepts = args.num_timepts
        self.point_net = PointNet(h_dim=args.pointnet_dim)

    def forward(self, batch, device):
        pts_msk_batch = batch["pts_msk_batch"].to(device)
        pts_tid_batch = batch["pts_tid_batch"].to(device)
        pts_aux_batch = batch["pts_aux_batch"].to(device)
        pts_cut_batch = batch["pts_cut_batch"]

        pts_obs_batch = batch["pts_obs_batch"]
        pts_obs_batch = Batch.from_data_list(pts_obs_batch)
        pts_obs_batch = pts_obs_batch.to(device)

        enc = self.point_net(
            pts_obs_batch.pos, pts_obs_batch.edge_index, pts_obs_batch.batch
        )
        enc = enc.tensor_split(pts_cut_batch, dim=0)
        enc = torch.stack(enc)

        N, T, D = enc.shape
        parts_inp_obs = torch.zeros(N, self.num_timepts, D, device=device)
        parts_inp_msk = torch.zeros(N, self.num_timepts, D, device=device)
        parts_inp_tps = torch.zeros(N, self.num_timepts, device=device)
        parts_inp_obs[:, :T] = enc
        parts_inp_tps[:, :T] = pts_tid_batch / self.num_timepts
        parts_inp_msk[:, :T] = 1
        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)

        pts_tid_batch = pts_tid_batch.view(
            pts_tid_batch.shape + torch.Size([1])
        ).expand_as(enc)
        evd_obs = torch.zeros(N, self.num_timepts, D, device=device)
        evd_obs.scatter_(1, pts_tid_batch, enc)
        evd_msk = pts_msk_batch.expand(N, self.num_timepts, D)

        return inp, evd_obs, evd_msk, pts_aux_batch


class JointBackbone(nn.Module):
    """Combines a TDABackone with a PointNet backbone."""
    def __init__(self, args):
        super(JointBackbone, self).__init__()
        self.num_timepts = args.num_timepts
        self.point_net = PointNet(h_dim=args.pointnet_dim)
        self.recog_net = VecRecogNet(
            mtan_input_dim=args.vec_inp_dim + args.pointnet_dim,
            mtan_hidden_dim=args.mtan_h_dim,
            latent_dim=args.z_dim,
            use_atanh=False,
        )

    def forward(self, batch, device):
        batch_tda = batch["tda_obs_batch"]
        parts = {key: val.to(device) for key, val in batch_tda.items()}
        parts_inp_obs = parts["inp_obs"]
        parts_inp_msk = parts["inp_msk"]
        parts_inp_tps = parts["inp_tps"]

        pts_aux_batch = batch["pts_aux_batch"].to(device)
        pts_tid_batch = batch["pts_tid_batch"].to(device)
        pts_cut_batch = batch["pts_cut_batch"]
        pts_obs_batch = batch["pts_obs_batch"]
        pts_obs_batch = Batch.from_data_list(pts_obs_batch)
        pts_obs_batch = pts_obs_batch.to(device)

        enc = self.point_net(
            pts_obs_batch.pos, pts_obs_batch.edge_index, pts_obs_batch.batch
        )
        enc = enc.tensor_split(pts_cut_batch, dim=0)
        enc = torch.stack(enc)

        N, T, D = enc.shape
        enc_ext = torch.zeros(N, self.num_timepts, D, device=device)
        enc_ext[:, :T] = enc

        parts_inp_obs = torch.cat((parts_inp_obs, enc_ext), dim=2)  # N,T,D
        parts_inp_msk = (
            parts_inp_msk[:, :, 0]
            .view(parts_inp_obs.shape[0], parts_inp_obs.shape[1], 1)
            .expand(
                parts_inp_obs.shape[0], parts_inp_obs.shape[1], parts_inp_obs.shape[2]
            )
        )

        pts_tid_batch = pts_tid_batch.view(
            pts_tid_batch.shape + torch.Size([1])
        ).expand_as(enc)
        evd_obs = torch.zeros(N, self.num_timepts, D, device=device)
        evd_obs.scatter_(1, pts_tid_batch, enc)
        evd_obs = torch.cat((evd_obs, parts["evd_obs"]), dim=2)
        evd_msk = (
            parts["evd_msk"][:, :, 0]
            .view(evd_obs.shape[0], evd_obs.shape[1], 1)
            .expand(evd_obs.shape[0], evd_obs.shape[1], evd_obs.shape[2])
        )

        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)
        return self.recog_net(inp), evd_obs, evd_msk, pts_aux_batch


class JointBare(nn.Module):
    def __init__(self, args):
        super(JointBare, self).__init__()
        self.point_net = PointNet(h_dim=args.pointnet_dim)

    def forward(self, batch, device):
        batch_tda = batch["tda_obs_batch"]
        parts = {key: val.to(device) for key, val in batch_tda.items()}
        parts_inp_obs = parts["inp_obs"]
        parts_inp_msk = parts["inp_msk"]
        parts_inp_tps = parts["inp_tps"]

        pts_aux_batch = batch["pts_aux_batch"].to(device)
        pts_tid_batch = batch["pts_tid_batch"].to(device)
        pts_cut_batch = batch["pts_cut_batch"]
        pts_obs_batch = batch["pts_obs_batch"]
        pts_obs_batch = Batch.from_data_list(pts_obs_batch)
        pts_obs_batch = pts_obs_batch.to(device)

        enc = self.point_net(
            pts_obs_batch.pos, pts_obs_batch.edge_index, pts_obs_batch.batch
        )
        enc = enc.tensor_split(pts_cut_batch, dim=0)
        enc = torch.stack(enc)

        N, T, D = enc.shape
        enc_ext = torch.zeros(N, self.num_timepts, D, device=device)
        enc_ext[:, :T] = enc

        parts_inp_obs = torch.cat((parts_inp_obs, enc_ext), dim=2)  # N,T,D
        parts_inp_msk = (
            parts_inp_msk[:, :, 0]
            .view(parts_inp_obs.shape[0], parts_inp_obs.shape[1], 1)
            .expand(
                parts_inp_obs.shape[0], parts_inp_obs.shape[1], parts_inp_obs.shape[2]
            )
        )

        pts_tid_batch = pts_tid_batch.view(
            pts_tid_batch.shape + torch.Size([1])
        ).expand_as(enc)
        evd_obs = torch.zeros(N, self.num_timepts, D, device=device)
        evd_obs.scatter_(1, pts_tid_batch, enc)
        evd_obs = torch.cat((evd_obs, parts["evd_obs"]), dim=2)
        evd_msk = (
            parts["evd_msk"][:, :, 0]
            .view(evd_obs.shape[0], evd_obs.shape[1], 1)
            .expand(evd_obs.shape[0], evd_obs.shape[1], evd_obs.shape[2])
        )

        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)
        return inp, evd_obs, evd_msk, pts_aux_batch

class MTANHead(nn.Module):
    """Implements the mTAN twin of the encoder. However, this one operates
    directly on latent state trajectories.
    """
    def __init__(
        self,
        mtan_input_dim: int = 32,
        mtan_hidden_dim: int = 32,
        num_timepts: int = 100,
        use_atanh: bool = False,
    ) -> None:
        super().__init__()
        self.mtan_input_dim = mtan_input_dim
        self.mtan_hidden_dim = mtan_hidden_dim
        self.num_timepts = num_timepts
        self.use_atanh = use_atanh
        self.learn_emb = True
        self.mtan = MTANEncoder(
            input_dim=mtan_input_dim,
            query=torch.linspace(0, 1.0, 128),
            nhidden=mtan_hidden_dim,
            embed_time=128,
            num_heads=4,
            learn_emb=self.learn_emb,
        )

    def get_outdim(self) -> int:
        return self.mtan_hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        observed_data = x
        observed_mask = torch.ones_like(x)
        observed_tp = (
            torch.linspace(0, 1.0, self.num_timepts, device=x.device)
            .unsqueeze(0)
            .expand(x.shape[0], -1)
        )
        h = self.mtan(torch.cat((observed_data, observed_mask), 2), observed_tp)
        if self.use_atanh:
            eps = 1e-5
            h = h - h.sign() * eps
            h = h.atanh()
        return h


def normal_kl(mu1: torch.Tensor, lv1: torch.Tensor, mu2: torch.Tensor, lv2: torch.Tensor):
    """Implements the KL divergence for normal distributions."""
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    return kl


class VecRecogNetBaseline(nn.Module):
    """Implements the mTAN baseline, i.e., directly predicting simulation parameters
    from the encoder output.
    """
    def __init__(
        self,
        mtan_input_dim: int = 32,
        mtan_hidden_dim: int = 32,
        mtan_embed_time: int = 128,
        mtan_num_queries: int = 128,
        use_atanh: bool = False,
    ) -> None:
        super().__init__()
        self.mtan_input_dim = mtan_input_dim
        self.mtan_hidden_dim = mtan_hidden_dim
        self.mtan_embed_time = mtan_embed_time
        self.mtan_num_queries = mtan_num_queries
        self.use_atanh = use_atanh
        self.learn_emb = True
        self.mtan = MTANEncoder(
            input_dim=mtan_input_dim,
            query=torch.linspace(0, 1.0, mtan_num_queries),
            nhidden=mtan_hidden_dim,
            embed_time=mtan_embed_time,
            num_heads=1,
            learn_emb=self.learn_emb,
        )

    def extra_repr(self) -> str:
        return (
            f"mtan_input_dim={self.mtan_hidden_dim}, "
            f"mtan_hidden_dim={self.mtan_hidden_dim}, use_atanh={self.use_atanh}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        observed_data, observed_mask, observed_tp = x
        h = self.mtan(torch.cat((observed_data, observed_mask), 2), observed_tp)
        if self.use_atanh:
            eps = 1e-5
            h = h - h.sign() * eps
            h = h.atanh()
        return h


class VecRecogNet(nn.Module):
    def __init__(
        self,
        mtan_input_dim: int = 32,
        mtan_hidden_dim: int = 32,
        latent_dim: int = 16,
        use_atanh: bool = False,
    ) -> None:
        super().__init__()
        self.mtan_input_dim = mtan_input_dim
        self.mtan_hidden_dim = mtan_hidden_dim
        self.use_atanh = use_atanh
        self.learn_emb = True
        self.mtan = MTANEncoder(
            input_dim=mtan_input_dim,
            query=torch.linspace(0, 1.0, 128),  # tested with 64 in ablation
            nhidden=mtan_hidden_dim,
            embed_time=128,  # tested with 64 in ablation
            num_heads=1,
            learn_emb=self.learn_emb,
        )

        self.h_to_z = nn.Linear(mtan_hidden_dim, latent_dim * 2)

    def extra_repr(self) -> str:
        return (
            f"mtan_input_dim={self.mtan_hidden_dim}, "
            f"mtan_hidden_dim={self.mtan_hidden_dim}, use_atanh={self.use_atanh}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        observed_data, observed_mask, observed_tp = x
        h = self.mtan(torch.cat((observed_data, observed_mask), 2), observed_tp)
        if self.use_atanh:
            eps = 1e-5
            h = h - h.sign() * eps
            h = h.atanh()
        return self.h_to_z(h)


class VecReconNet(nn.Module):
    """Implements the reconstruction network via a simple MLP."""
    def __init__(self, z_dim: int=16, h_dim: int=32, x_dim: int=10) -> None:
        super(VecReconNet, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.map = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Linear(in_features=h_dim, out_features=x_dim),
        )

    def forward(self, z):
        return self.map(z)


# taken from torchdiffeq (repo) examples
class LatentODEfunc(nn.Module):
    def __init__(self, z_dim=4, h_dim=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class PathToGaussianDecoder(nn.Module):
    def __init__(
        self,
        mu_map: nn.Module,
        sigma_map: Optional[nn.Module] = None,
        initial_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.mu_map = mu_map
        self.sigma_map = sigma_map
        self.initial_sigma = initial_sigma
        if self.sigma_map is None:
            self.sigma = nn.Parameter(torch.tensor(initial_sigma))

    def extra_repr(self) -> str:
        if self.sigma_map is None:
            s = f"initial_sigma={self.initial_sigma}"
        else:
            s = ""
        return s

    def forward(self, x: torch.Tensor) -> Normal:
        n_samples, batch_size, time_steps, _ = x.shape
        target_shape = [1 for i in range(len(x.shape))]

        mu = self.mu_map(x.flatten(0, 2))
        mu = mu.unflatten(0, (n_samples, batch_size, time_steps))

        if self.sigma_map is not None:
            sigma = self.sigma_map(x)
        else:
            sigma = self.sigma.view(target_shape).expand_as(mu)
        return Normal(mu, sigma.square())


# region mTAN core
class MultiTimeAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhidden: int = 16,
        embed_time: int = 16,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        assert embed_time % num_heads == 0
        self.input_dim = input_dim
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.embed_time_k = embed_time // num_heads
        self.nhidden = nhidden
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def extra_repr(self) -> str:
        return (
            "input_dim={self.input_dim}, nhidden={self.nhidden}, "
            f"embed_time={self.embed_time}, num_heads={self.num_heads})"
        )

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> torch.Tensor:
        batch, _, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [
            l(x).view(x.size(0), -1, self.num_heads, self.embed_time_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.num_heads * dim)
        return self.linears[-1](x)


class EncMtanRnn(nn.Module):
    def __init__(
        self,
        input_dim: int,
        query: torch.Tensor,
        latent_dim: int = 2,
        nhidden: int = 16,
        embed_time: int = 16,
        num_heads: int = 1,
        learn_emb: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim  # self.dim
        self.register_buffer("query", query)
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.learn_emb = learn_emb
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * nhidden, 50), nn.ReLU(), nn.Linear(50, latent_dim * 2)
        )
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time - 1)
            self.linear = nn.Linear(1, 1)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, query=Tensor: {self.query.shape}, "
            f"latent_dim={self.latent_dim}, nhidden={self.nhidden}, "
            f"embed_time={self.embed_time}, num_heads={self.num_heads}, "
            f"learn_emb={self.learn_emb})"
        )

    def learn_time_embedding(self, tt: torch.Tensor) -> torch.Tensor:
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos: torch.Tensor) -> torch.Tensor:
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.0 * pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        time_steps = time_steps
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        out = self.hiddens_to_z0(out)
        return out


class MTANEncoder(EncMtanRnn):
    def __init__(
        self,
        input_dim: int,
        query: torch.Tensor,
        latent_dim: int = 2,
        nhidden: int = 16,
        embed_time: int = 16,
        num_heads: int = 1,
        learn_emb: bool = False,
    ) -> None:
        super().__init__(
            input_dim, query, latent_dim, nhidden, embed_time, num_heads, learn_emb
        )
        self.input_dim = input_dim
        self.query = query
        self.latent_dim = latent_dim
        self.nhidden = nhidden
        self.embed_time = embed_time
        self.num_heads = num_heads
        self.learn_emb = learn_emb
        self.hiddens_to_z0 = None
        self.gru_rnn = nn.GRU(nhidden, nhidden, bidirectional=False, batch_first=True)

    def __repr__(self) -> str:
        return (
            f"MTANEncoder(input_dim={self.input_dim}, query={self.query}, "
            f"latent_dim={self.latent_dim}, nhidden={self.nhidden}, "
            f"embed_time={self.embed_time}, num_heads={self.num_heads}, "
            f"learn_emb={self.learn_emb})"
        )

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        mask = x[:, :, self.input_dim :]
        mask = torch.cat((mask, mask), 2)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(self.query.unsqueeze(0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, x, mask)
        _, out = self.gru_rnn(out)
        return out.squeeze(0)
# endregion


# region Baseline backbones
class TDABaselineBackbone(nn.Module):
    def __init__(self, args):
        super(TDABaselineBackbone, self).__init__()
        self.num_timepts = args.num_timepts    
        self.recog_net = VecRecogNetBaseline(
            mtan_input_dim=args.vec_inp_dim, 
            mtan_hidden_dim=args.mtan_h_dim, 
            mtan_embed_time=args.mtan_embed_time,
            mtan_num_queries=args.mtan_num_queries,
            use_atanh=False)
    
    def forward(self, batch, device):
        parts = {key: val.to(device) for key, val in batch.items()}
        parts_inp_obs = parts['inp_obs']
        parts_inp_msk = parts['inp_msk']
        parts_inp_tps = parts['inp_tps']  
        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)
        return self.recog_net(inp), parts['evd_obs'], parts['evd_msk'], parts['aux_obs']


class PointNetBaselineBackbone(nn.Module):
    def __init__(self, args):
        super(PointNetBaselineBackbone, self).__init__()
        self.num_timepts = args.num_timepts
        self.point_net = PointNet(h_dim=args.pointnet_dim)
        self.recog_net = VecRecogNetBaseline(
            mtan_input_dim=args.pointnet_dim, 
            mtan_hidden_dim=args.mtan_h_dim, 
            use_atanh=False)

    def forward(self, batch, device):
        pts_msk_batch = batch['pts_msk_batch'].to(device)
        pts_tid_batch = batch['pts_tid_batch'].to(device)  
        pts_aux_batch = batch['pts_aux_batch'].to(device)
        pts_cut_batch = batch['pts_cut_batch']
        
        pts_obs_batch = batch['pts_obs_batch']     
        pts_obs_batch = Batch.from_data_list(pts_obs_batch)
        pts_obs_batch = pts_obs_batch.to(device)
    
        enc = self.point_net(pts_obs_batch.pos, pts_obs_batch.edge_index, pts_obs_batch.batch)
        enc = enc.tensor_split(pts_cut_batch, dim=0)
        enc = torch.stack(enc) 
        
        N,T,D = enc.shape
        parts_inp_obs = torch.zeros(N,self.num_timepts,D,device=device)
        parts_inp_msk = torch.zeros(N,self.num_timepts,D,device=device)
        parts_inp_tps = torch.zeros(N,self.num_timepts,device=device)
        parts_inp_obs[:,:T] = enc
        parts_inp_tps[:,:T] = pts_tid_batch/self.num_timepts
        parts_inp_msk[:,:T] = 1
        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)
        
        pts_tid_batch = pts_tid_batch.view(pts_tid_batch.shape + torch.Size([1])).expand_as(enc)
        evd_obs = torch.zeros(N,self.num_timepts,D,device=device)
        evd_obs.scatter_(1,pts_tid_batch,enc)
        evd_msk = pts_msk_batch.expand(N,self.num_timepts,D)
    
        return self.recog_net(inp), evd_obs, evd_msk, pts_aux_batch


class JointBaselineBackbone(nn.Module):
    def __init__(self, args):
        super(JointBaselineBackbone, self).__init__()
        self.num_timepts = args.num_timepts    
        self.point_net = PointNet(h_dim=args.pointnet_dim)
        self.recog_net = VecRecogNetBaseline(
            mtan_input_dim=args.vec_inp_dim + args.pointnet_dim, 
            mtan_hidden_dim=args.mtan_h_dim, 
            use_atanh=False)
    
    def forward(self, batch, device):
        batch_tda = batch['tda_obs_batch']        
        parts = {key: val.to(device) for key, val in batch_tda.items()}
        parts_inp_obs = parts['inp_obs']
        parts_inp_msk = parts['inp_msk']
        parts_inp_tps = parts['inp_tps']    
        
        pts_aux_batch = batch['pts_aux_batch'].to(device)  
        pts_tid_batch = batch['pts_tid_batch'].to(device)  
        pts_cut_batch = batch['pts_cut_batch']          
        pts_obs_batch = batch['pts_obs_batch']     
        pts_obs_batch = Batch.from_data_list(pts_obs_batch)
        pts_obs_batch = pts_obs_batch.to(device)
    
        enc = self.point_net(pts_obs_batch.pos, pts_obs_batch.edge_index, pts_obs_batch.batch)
        enc = enc.tensor_split(pts_cut_batch, dim=0)
        enc = torch.stack(enc) 
        
        N,T,D = enc.shape
        enc_ext = torch.zeros(N,self.num_timepts,D,device=device)
        enc_ext[:,:T] = enc
        
        parts_inp_obs = torch.cat((parts_inp_obs, enc_ext), dim=2)  # N,T,D
        parts_inp_msk = parts_inp_msk[:,:,0].view(
            parts_inp_obs.shape[0],
            parts_inp_obs.shape[1],1).expand(
                parts_inp_obs.shape[0],
                parts_inp_obs.shape[1],
                parts_inp_obs.shape[2])
                
        pts_tid_batch = pts_tid_batch.view(pts_tid_batch.shape + torch.Size([1])).expand_as(enc)
        evd_obs = torch.zeros(N,self.num_timepts,D,device=device)
        evd_obs.scatter_(1,pts_tid_batch,enc)
        evd_obs = torch.cat((evd_obs, parts['evd_obs']),dim=2)
        evd_msk = parts['evd_msk'][:,:,0].view(evd_obs.shape[0],evd_obs.shape[1],1).expand(
            evd_obs.shape[0],
            evd_obs.shape[1],
            evd_obs.shape[2])
        
        inp = (parts_inp_obs, parts_inp_msk, parts_inp_tps)
        return self.recog_net(inp), evd_obs, evd_msk, pts_aux_batch
# endregion