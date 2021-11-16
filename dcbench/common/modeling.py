from abc import abstractmethod

import PIL
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
from torchvision.models import DenseNet as _DenseNet
from torchvision.models import ResNet as _ResNet
from torchvision.models.densenet import _load_state_dict
from torchvision.models.densenet import model_urls as densenet_model_urls
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls as resnet_model_urls


class Model(pl.LightningModule):

    DEFAULT_CONFIG = {}

    def __init__(self, config: dict = None):
        super().__init__()
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            self.config.update(config)

        self._set_model()

    @abstractmethod
    def _set_model(self):
        raise NotImplementedError()


class ResNet(_ResNet):

    ACTIVATION_DIMS = [64, 128, 256, 512]
    ACTIVATION_WIDTH_HEIGHT = [64, 32, 16, 8]
    RESNET_TO_ARCH = {"resnet18": [2, 2, 2, 2], "resnet50": [3, 4, 6, 3]}

    def __init__(
        self,
        num_classes: int,
        arch: str = "resnet18",
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        if arch not in self.RESNET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.RESNET_TO_ARCH.keys()}"
            )

        block = BasicBlock if arch == "resnet18" else Bottleneck
        super().__init__(block, self.RESNET_TO_ARCH[arch])
        if pretrained:
            state_dict = load_state_dict_from_url(
                resnet_model_urls[arch], progress=True
            )
            self.load_state_dict(state_dict)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512 * block.expansion, num_classes)
        )


def default_transform(img: PIL.Image.Image):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(img)


def default_train_transform(img: PIL.Image.Image):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(img)


class DenseNet(_DenseNet):

    DENSENET_TO_ARCH = {
        "densenet121": {
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
            "num_init_features": 64,
        }
    }

    def __init__(
        self, num_classes: int, arch: str = "densenet121", pretrained: bool = True
    ):
        if arch not in self.DENSENET_TO_ARCH:
            raise ValueError(
                f"config['classifier'] must be one of: {self.DENSENET_TO_ARCH.keys()}"
            )

        super().__init__(**self.DENSENET_TO_ARCH[arch])
        if pretrained:
            _load_state_dict(self, densenet_model_urls[arch], progress=True)

        self.classifier = nn.Linear(self.classifier.in_features, num_classes)


class VisionClassifier(Model):

    DEFAULT_CONFIG = {
        "lr": 1e-4,
        "model_name": "resnet",
        "arch": "resnet18",
        "pretrained": True,
        "num_classes": 2,
        "transform": default_transform,
        "train_transform": default_train_transform,
    }

    def _set_model(self):
        if self.config["model_name"] == "resnet":
            self.model = ResNet(
                num_classes=self.config["num_classes"],
                arch=self.config["arch"],
                pretrained=self.config["pretrained"],
            )
        elif self.config["model_name"] == "densenet":
            self.model = DenseNet(
                num_classes=self.config["num_classes"], arch=self.config["arch"]
            )
        else:
            raise ValueError(f"Model name {self.config['model_name']} not supported.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch["input"], batch["target"], batch["id"]
        outs = self.forward(inputs)

        loss = nn.functional.cross_entropy(outs, targets)
        self.log("train_loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch["input"], batch["target"]

        outs = self.forward(inputs)
        loss = nn.functional.cross_entropy(outs, targets)
        self.log("valid_loss", loss)

    def validation_epoch_end(self, outputs) -> None:
        for metric_name, metric in self.metrics.items():
            self.log(f"valid_{metric_name}", metric.compute())
            metric.reset()

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return optimizer
