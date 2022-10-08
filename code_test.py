import yaml


def test_config():
    config_path = 'configs/operator/Re500-PINO.yaml'

    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    print(config['data']['paths'])


if __name__ == '__main__':
    test_config()
