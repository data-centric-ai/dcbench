import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED = [
    "click>=8.0.0",
    "pyyaml>=5.4",
    "pre-commit",
    # "pytorch-lightning",
    "pandas",
    "numpy>=1.18.0",
    # "cytoolz",
    "ujson",
    "jsonlines>=1.2.0",
    # "torch>=1.8.0",
    "tqdm>=4.49.0",
    "scikit-learn",
    # "torchvision>=0.9.0",
    # "wandb",
    # "ray[default]",
    # `"torchxrayvision",`
]

setuptools.setup(
    name="dcbench",
    version="0.0.1",
    description="Data-centric AI benchmark.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HazyResearch/data-centric-benchmark",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": ["dcbench=dcbench:main"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=REQUIRED,
)
