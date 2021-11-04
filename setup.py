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
EXTRAS = {
    "dev": [
        "black>=21.5b0",
        "isort>=5.7.0",
        "flake8>=3.8.4",
        "docformatter>=1.4",
        "pytest-cov>=2.10.1",
        "sphinx-rtd-theme>=0.5.1",
        "nbsphinx>=0.8.0",
        "recommonmark>=0.7.1",
        "parameterized",
        "pre-commit>=2.9.3",
        "sphinx-autobuild",
    ],
}

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
    extras_require=EXTRAS,
)
