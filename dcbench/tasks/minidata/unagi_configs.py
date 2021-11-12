MIXER_MODEL = {
    "name": "mixer",
    "train": True,
    "d": 256,
    "num_layers": 7,
    "batch_size": 16,
    "mlp_dim": 512,
    "patch_size": 4,
    "dropout": 0.3,
    "patch_emb_type": "square",
}

TRANFORMER_MODEL = {
    "type": "patchnet",
    "patch_emb_type": "square",
    "patch_size": 8,
    "d": 256,
    "grayscale": False,
    "learn_pos": True,
    "num_heads": 8,
    "num_layers": 7,
    "dropout": 0.05,
    "head_dropout": 0.1,
    "use_cls_token": True,
    "use_all_tokens": False,
}

# TODO(sabri): add resnet

DEFAULT_UNAGI_CONFIG = {
    "model": MIXER_MODEL,
    "augmentations": {
        "raw": {
            "image_pil": [
                {"type": "RandomResizeCrop", "params": {"prob": 1.0, "size": 32}},
                {"type": "HorizontalFlip", "params": {"prob": 0.5}},
                {"type": "ColorDistortion", "params": {"prob": 1.0}},
                {"type": "GaussianBlur", "params": {"prob": 0.5, "kernel_size": 3}},
            ]
        },
        "patch": {"type": None},
        "feature": {"type": None},
    },
    "tasks": {
        "supervised": {"loss_fn": "cross_entropy"},
    },
    "dataset": {
        "name": "cifar10",
        "task": "multi_class",
        "input_features": [
            {"name": "image_pil", "type": "image", "augmentation": "image_pil"}
        ],
        "output_features": [{"name": "labels"}],
    },
    "learner_config": {
        "n_epochs": 2,
        "optimizer_config": {"optimizer": "adamw", "lr": 0.001},
        "lr_scheduler_config": {
            "lr_scheduler": "plateau",
            "lr_scheduler_step_unit": "epoch",
            "plateau_config": {"factor": 0.2, "patience": 10, "threshold": 0.01},
        },
    },
}
