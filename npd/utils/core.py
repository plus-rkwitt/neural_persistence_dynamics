import torch

def compute_minmax_reverse_stats(prms_orig_file):
    prms_orig = torch.load(prms_orig_file)
    max_d = prms_orig.max(dim=0, keepdim=True).values
    min_d = prms_orig.min(dim=0, keepdim=True).values
    return min_d, max_d

def compute_minmax_reverse(x, min_d, max_d):
    return (x + 1.)/2 * (max_d-min_d) + min_d