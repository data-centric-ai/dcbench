import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_ENV_VARIABLE = "DCBENCH_CONFIG"
default_local_dir = os.path.join(Path.home(), ".dcbench")


def get_config_location():
    path = os.environ.get(
        CONFIG_ENV_VARIABLE, os.path.join(default_local_dir, "dcbench-config.yaml")
    )
    return path


def get_config():
    path = get_config_location()
    if not os.path.exists(path):
        config = {}
    else:
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    return config


@dataclass
class DCBenchConfig:

    local_dir: str = default_local_dir
    public_bucket_name: str = "dcbench"
    hidden_bucket_name: str = "dcbench-hidden"

    @property
    def public_remote_url(self):
        return f"https://storage.googleapis.com/{self.public_bucket_name}"

    @property
    def hidden_remote_url(self):
        return f"https://storage.googleapis.com/{self.hidden_bucket_name}"

    # dataset specific download directories
    celeba_dir: str = os.path.join(default_local_dir, "datasets", "celeba")
    imagenet_dir: str = os.path.join(default_local_dir, "datasets", "imagenet")


config = DCBenchConfig(**get_config())
