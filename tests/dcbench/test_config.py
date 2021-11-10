import os

import yaml

from dcbench.config import DCBenchConfig, get_config


def test_env(tmpdir):
    config_path = os.path.join(tmpdir, "config.yaml")

    new_local_dir = os.path.join(tmpdir, ".dcbench-env")

    yaml.dump({"local_dir": new_local_dir}, open(config_path, "w"))
    os.environ["DCBENCH_CONFIG"] = config_path

    config = DCBenchConfig(**get_config())
    assert config.local_dir == new_local_dir
