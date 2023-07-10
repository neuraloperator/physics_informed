import os
from argparse import ArgumentParser
import requests
from tqdm import tqdm


_url_dict = {
    'NS-T4000': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fft_Re500_T4000.npy', 
    'NS-Re500Part0': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fine_Re500_T128_part0.npy', 
    'NS-Re500Part1': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fine_Re500_T128_part1.npy', 
    'NS-Re500Part2': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fine_Re500_T128_part2.npy', 
    'NS-Re100Part0': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fine_Re100_T128_part0.npy', 
    'burgers': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/burgers_pino.mat', 
    'NS-Re500_T300_id0': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/NS-Re500_T300_id0.npy',
    'NS-Re500_T300_id0-shuffle': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS-Re500_T300_id0-shuffle.npy',
    'darcy-train': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/piececonst_r421_N1024_smooth1.mat', 
    'darcy-test': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/piececonst_r421_N1024_smooth2.mat', 
    'cavity': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/cavity.mat',
    'Re500-1_8s-800-pino-140k': 'https://hkzdata.s3.us-west-2.amazonaws.com/PINO/checkpoints/Re500-1_8s-800-PINO-140000.pt',
}


def download_file(url, file_path):
    print('Start downloading...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=256 * 1024 * 1024)):
                f.write(chunk)
    print('Complete')


def main(args):
    url = _url_dict[args.name]
    file_name = url.split('/')[-1]
    os.makedirs(args.outdir, exist_ok=True)
    file_path = os.path.join(args.outdir, file_name)
    download_file(url, file_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='Parser for downloading assets')
    parser.add_argument('--name', type=str, default='NS-T4000')
    parser.add_argument('--outdir', type=str, default='../data')
    args = parser.parse_args()
    main(args)