import os
import yaml

class ConfigLoader:

    def __init__(self, config_file_path="config.yaml"):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(self.base_dir, config_file_path)

        with open(full_path, "r") as yml_file:
            self.config = yaml.safe_load(yml_file)


    def get_path(self, relative_path: str) -> str:
        return os.path.join(self.base_dir, relative_path)
