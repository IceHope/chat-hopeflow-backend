from pathlib import Path


class BaseConfiguration:
    """
    Base class for configuration
    """

    def __init__(self):
        import toml
        path = str(Path(__file__).parent / "base_settings.toml")
        # 打开并读取 TOML 文件
        with open(path, 'r', encoding="utf-8") as file:
            self.config = toml.load(file)

    def get_redis_config(self):
        return self.config['redis']

    def get_username_admin_test(self):
        return self.config['username']['admin_test']


if __name__ == '__main__':
    config = BaseConfiguration().get_username_admin_test()
    print(config)
