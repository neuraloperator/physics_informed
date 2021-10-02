import yaml
from argparse import ArgumentParser
from baselines.train import train_deeponet_cp
from baselines.test import test_deeponet_cp


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='train', help='Train or test')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.mode == 'train':
        print('Start training DeepONet Cartesian Product')
        train_deeponet_cp(config)
    else:
        print('Start testing DeepONet Cartesian Product')
        test_deeponet_cp(config)
    print('Done!')