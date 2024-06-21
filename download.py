"""Download precomputed simulation data, persistence diagrams and vectorizations.

szeng, fgraf, muray, shuber, 2024
"""

import os
import gdown
import argparse
from pathlib import Path


DOWNLOAD_LINKS = {
    'dorsogna-1k': {
        'simu_1k.pt': '1h3Dz9uM5lN2q3MEoqjNQZKOm8Vtqevcu',
        'prms_1k.pt': '17lKLfqxp7gjkJJR9YSNeDKeiat6FNPxN',
        'prms_1k_norm.pt': '1ZiMM2ALYkM7DnXaTHWyr3qUjcQrzMj8O',
        'dgms_1k_vr_h0h1.pt': '15J3XbWjbG3WZQgH-2UbxjUTKl5rDpLmx',
        'vecs_20_0.005.pt': '1QKzw1TjWlJ3sjX1Iy7akcF5iwxAleR3n',
        'dgms_1k_vr_h0h1.pt': '15J3XbWjbG3WZQgH-2UbxjUTKl5rDpLmx'
    },
    'volex-10k': {
        'simu_10k.pt': '1pevrjzcL1lLDeDgFh1uNOXlW1V-Qb6fo',
        'prms_1k.pt': '1crOSVmiMkoHyWrHH7RqIjITUN1q506s_',
        'prms_1k_norm.pt': '1kFR54X-8IP-Ca2ljd0oCyx7MfvSWi1p2',
        'dgms_1k_vr_h0h1.pt': '1OKfdKVrmzAGcMacycuBpfcHriiEoeVRo',    
        'vecs_20_0.005.pt': '1UaMhNBeFtqlX12WdAp_oXuPpTQq8nHbs'
    }
}


def setup_cmdline_parsing():
    generic_parser = argparse.ArgumentParser()
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument("--dataset", 
                        type=str, 
                        default='dorsogna-1k',
                        choices=['dorsogna-1k', 'volex-10k'])
    group0.add_argument("--destination", type=str, default="/tmp/")
    return generic_parser

def main():
    
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(args)
    
    Path(args.destination).mkdir(parents=True, exist_ok=True)
    
    links = DOWNLOAD_LINKS[args.dataset]
    
    for k,v in links.items():
        url = 'https://drive.google.com/uc?id={}'.format(v)
        output = os.path.join(args.destination,k)
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)
        else:
            print(f'{output} already exists!')

if __name__ == "__main__":
    main()


