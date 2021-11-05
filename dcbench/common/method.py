from abc import ABC
from dataclasses import dataclass


class Method(ABC):
    @dataclass
    class Config:
        n_slices: int = 5
        emb_group: str = "main"
        emb: str = "emb"
        xmodal_emb: str = "emb"

    RESOURCES_REQUIRED = {"cpu": 1, "custom_resources": {"ram_gb": 4}}

    def __init__(self, config: dict = None, **kwargs):
        if config is not None:
            self.config = self.Config(**config, **kwargs)
        else:
            self.config = self.Config(**kwargs)
