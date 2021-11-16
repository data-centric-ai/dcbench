MIXER_CONFIG = {
    "model": {
        "name": "mixer",
        "train": True,
        "d": 256,
        "num_heads": 8,
        "head_dropout": 0.1,
        "label_smoothing": True,
        "mlp_dim": 512,
        "num_layers": 7,
        "patch_size": 4,
        "dropout": 0.05,
        "max_sequence_length": 64,
        "patch_emb_type": "square",
    },
    "augmentations": {
        "raw": {
            "image_pil": [
                {"type": "RandomResizeCrop", "params": {"prob": 1.0, "size": 32}},
                {"type": "HorizontalFlip", "params": {"prob": 0.5}},
                {"type": "ColorDistortion", "params": {"prob": 1.0}},
                {"type": "GaussianBlur", "params": {"prob": 0.5, "kernel_size": 3}},
            ],
            "image_pil_default_transform_transformer": [
                {"type": "ToTensor"},
                {
                    "type": "Normalize",
                    "params": {
                        "mean": [0.49139968, 0.48215841, 0.44653091],
                        "std": [0.24703223, 0.24348513, 0.26158784],
                    },
                },
                {"type": "Reshape2D", "params": {"h_dim": 3, "w_dim": 1024}},
            ],
            "image_pil_default_transform_resnet": [
                {"type": "ToTensor"},
                {
                    "type": "Normalize",
                    "params": {
                        "mean": [0.49139968, 0.48215841, 0.44653091],
                        "std": [0.24703223, 0.24348513, 0.26158784],
                    },
                },
            ],
        },
        "patch": {"type": None},
        "feature": {"type": None},
    },
    "tasks": {"supervised": {"loss_fn": "cross_entropy", "label": "target"}},
    "dataset": {
        "name": "meerkat_dataset",
        "meerkat_dataset_name": None,
        "index_name": "id",
        "task": "multi_class",
        "input_features": [
            {
                "name": "image",
                "type": "image",
                "transformation": "image_pil",
                "default_transformation": "image_pil_default_transform_transformer",
            }
        ],
        "output_features": [{"name": "target"}],
        "path_to_dp": None,
        "batch_size": 128,
        "val_batch_size": 128,
        "num_workers": 4,
    },
    "learner_config": {
        "n_epochs": 2,
        "train_split": "train",
        "valid_split": None,
        "test_split": "test",
        "optimizer_config": {"optimizer": "adamw", "lr": 0.001},
        "lr_scheduler_config": {
            "lr_scheduler": "plateau",
            "lr_scheduler_step_unit": "epoch",
            "plateau_config": {"factor": 0.2, "patience": 10, "threshold": 0.01},
        },
    },
}

TRANSFORMER_CONFIG = {
    "model": {
        "name": "patchnet",
        "train": True,
        "d": 256,
        "mlp_dim": 512,
        "num_layers": 7,
        "patch_size": 4,
        "grayscale": False,
        "num_heads": 8,
        "head_dropout": 0.1,
        "label_smoothing": True,
        "dropout": 0.05,
        "tie_weights": False,
        "learn_pos": True,
        "use_cls_token": True,
        "use_all_tokens": False,
        "max_sequence_length": 65,
        "patch_emb_type": "square",
    },
    "augmentations": {
        "raw": {
            "image_pil": [
                {"type": "RandomResizeCrop", "params": {"prob": 1.0, "size": 32}},
                {"type": "HorizontalFlip", "params": {"prob": 0.5}},
                {"type": "ColorDistortion", "params": {"prob": 1.0}},
                {"type": "GaussianBlur", "params": {"prob": 0.5, "kernel_size": 3}},
            ],
            "image_pil_default_transform_transformer": [
                {"type": "ToTensor"},
                {
                    "type": "Normalize",
                    "params": {
                        "mean": [0.49139968, 0.48215841, 0.44653091],
                        "std": [0.24703223, 0.24348513, 0.26158784],
                    },
                },
                {"type": "Reshape2D", "params": {"h_dim": 3, "w_dim": 1024}},
            ],
            "image_pil_default_transform_resnet": [
                {"type": "ToTensor"},
                {
                    "type": "Normalize",
                    "params": {
                        "mean": [0.49139968, 0.48215841, 0.44653091],
                        "std": [0.24703223, 0.24348513, 0.26158784],
                    },
                },
            ],
        },
        "patch": {"type": None},
        "feature": {"type": None},
    },
    "tasks": {"supervised": {"loss_fn": "cross_entropy", "label": "target"}},
    "dataset": {
        "name": "meerkat_dataset",
        "meerkat_dataset_name": None,
        "index_name": "id",
        "task": "multi_class",
        "batch_size": 128,
        "val_batch_size": 128,
        "num_workers": 4,
        "input_features": [
            {
                "name": "image",
                "type": "image",
                "transformation": "image_pil",
                "default_transformation": "image_pil_default_transform_transformer",
            }
        ],
        "output_features": [{"name": "target"}],
        "path_to_dp": None,
    },
    "learner_config": {
        "n_epochs": 2,
        "train_split": "train",
        "valid_split": None,
        "test_split": "test",
        "optimizer_config": {"optimizer": "adamw", "lr": 0.001},
        "lr_scheduler_config": {
            "lr_scheduler": "plateau",
            "lr_scheduler_step_unit": "epoch",
            "plateau_config": {"factor": 0.2, "patience": 10, "threshold": 0.01},
        },
    },
}

RESNET_CONFIG = {
    "model": {
        "name": "resnet",
        "train": True,
        "decoder_hidden_dim": 512,
        "decoder_projection_dim": 512,
        "model": "resnet50",
    },
    "augmentations": {
        "raw": {
            "image_pil": [
                {"type": "RandomResizeCrop", "params": {"prob": 1.0, "size": 32}},
                {"type": "HorizontalFlip", "params": {"prob": 0.5}},
                {"type": "ColorDistortion", "params": {"prob": 1.0}},
                {"type": "GaussianBlur", "params": {"prob": 0.5, "kernel_size": 3}},
            ],
            "image_pil_default_transform_transformer": [
                {"type": "ToTensor"},
                {
                    "type": "Normalize",
                    "params": {
                        "mean": [0.49139968, 0.48215841, 0.44653091],
                        "std": [0.24703223, 0.24348513, 0.26158784],
                    },
                },
                {"type": "Reshape2D", "params": {"h_dim": 3, "w_dim": 1024}},
            ],
            "image_pil_default_transform_resnet": [
                {"type": "ToTensor"},
                {
                    "type": "Normalize",
                    "params": {
                        "mean": [0.49139968, 0.48215841, 0.44653091],
                        "std": [0.24703223, 0.24348513, 0.26158784],
                    },
                },
            ],
        },
        "patch": {"type": None},
        "feature": {"type": None},
    },
    "tasks": {"supervised": {"loss_fn": "cross_entropy", "label": "target"}},
    "dataset": {
        "name": "meerkat_dataset",
        "meerkat_dataset_name": None,
        "index_name": "id",
        "task": "multi_class",
        "input_features": [
            {
                "name": "image",
                "type": "image",
                "transformation": "image_pil",
                "default_transformation": "image_pil_default_transform_resnet",
            }
        ],
        "output_features": [{"name": "target"}],
        "path_to_dp": None,
        "batch_size": 128,
        "val_batch_size": 128,
        "num_workers": 4,
    },
    "learner_config": {
        "n_epochs": 2,
        "train_split": "train",
        "valid_split": None,
        "test_split": "test",
        "optimizer_config": {"optimizer": "adamw", "lr": 0.001},
        "lr_scheduler_config": {
            "lr_scheduler": "plateau",
            "lr_scheduler_step_unit": "epoch",
            "plateau_config": {"factor": 0.2, "patience": 10, "threshold": 0.01},
        },
    },
}
