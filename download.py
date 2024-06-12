import os
import gdown
import argparse

DOWNLOAD_LINKS = {
    'dorsogna-1k': {
        'simu_1k.pt': '1h3Dz9uM5lN2q3MEoqjNQZKOm8Vtqevcu',
        'prms_1k.pt': '17lKLfqxp7gjkJJR9YSNeDKeiat6FNPxN',
        'prms_1k_norm.pt': '1ZiMM2ALYkM7DnXaTHWyr3qUjcQrzMj8O'
        }
    }


def setup_cmdline_parsing():
    generic_parser = argparse.ArgumentParser()
    group0 = generic_parser.add_argument_group('Data loading/saving arguments')
    group0.add_argument("--dataset", 
                        type=str, 
                        default='dorsogna-1k',
                        choices=['dorsogna-1k'])
    group0.add_argument("--destination", type=str, default="/tmp/")
    return generic_parser

def main():
    
    parser = setup_cmdline_parsing()
    args = parser.parse_args()
    print(args)
    
    links = DOWNLOAD_LINKS[args.dataset]
    
    for k,v in links.items():
        url = 'https://drive.google.com/uc?id={}'.format(v)
        output = os.path.join(args.destination,k)
        gdown.download(url, output, quiet=False)


if __name__ == "__main__":
    main()


