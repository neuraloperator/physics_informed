from argparse import ArgumentParser
import yaml

from baselines.pinns_ns_05s import train
from baselines.pinns_ns_50s import train_longtime
import csv


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--long', action='store_true', help='path to save log')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--stop', type=int, default=1, help='stopping index')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    with open(config['log']['logfile'], 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Error in u', 'Error in v', 'Error in w', 'Step'])

    for i in range(args.start, args.stop):
        print(f'Start to solve instance {i}')
        if 'time_scale' in config['data']:
            train_longtime(i, config, args)
        else:
            train(i, config, args)


