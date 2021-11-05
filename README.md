
<div align="center">
    <img src="docs/banner.png" height=200 alt="banner"/>
</div>

-----
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/data-centric-ai/dcbench/CI)
![GitHub](https://img.shields.io/github/license/data-centric-ai/dcbench)
[![Documentation Status](https://readthedocs.org/projects/dcbench/badge/?version=latest)](https://dcbench.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/data-centric-ai/dcbench/branch/main/graph/badge.svg?token=MOLQYUSYQU)](https://codecov.io/gh/data-centric-ai/dcbench)

dcbench tests various data-centric aspects of improving the quality of machine learning workflows.

[**Getting Started**](⚡️-Quickstart)
| [**What is Meerkat?**](💡-what-is-Meerkat)
| [**Docs**](https://meerkat.readthedocs.io/en/latest/index.html)
| [**Contributing**](CONTRIBUTING.md)
| [**Blogpost**](https://www.notion.so/sabrieyuboglu/Meerkat-DataPanels-for-Machine-Learning-64891aca2c584f1889eb0129bb747863)
| [**About**](✉️-About)

This is a benchmark that tests various data-centric aspects of improving the quality of machine learning workflows.

It features a growing list of *tasks*:

* Minimal data cleaning (`miniclean`)
* Task-specific Label Correction (`labelfix`)
* Discovery of validation Error Modalities (`errmod`)
* Minimal training dataset selection (`minitrain`)

Each task features a collection of *scenarios* which are defined by datasets and ML pipeline elements (e.g. a model, feature pre-processors, etc.)

## Basic Usage

The very first step is to install the PyPI package:

```bash
pip install dcai
```

Then, we advise using Jupyter notebooks or some other interactive environment. You start off by importing the library and listing all the available artefacts:

```python
from dcai import scenarios

scenarios.list()
```

You can then load a specific scenario and view its *artefacts*:

```python
scenario = scenarios.get("miniclean/bank")
scenario.artefacts
```

In the above example we are loading the `bank` scenario of the `miniclean` task. We can then load all the artefacts into a dictionary:

```python
a = scenario.artefacts.load()
```

This automatically downloads all the available artefacts, saves a local copy and loads it into memory. Artefacts can be accessed directly from the dictionary. We can then go ahead and write the code that will provide us with a scenario-specific solution:

```python
model.fit(a["X_train_dirty"], a["y_train"])

X_train_selection = ...
```

Once we have an object (e.g. `X_train_selection`) containing the scenario-specific solution, we can package it into a solution object:

```python
solution = scenario.solve(X_train_selection=X_train_selection)
```

We can then perform an evaluation on that solution that will give us the result:

```python
solution.evaluate()
solution.result
```

After you're happy with the obtained result, you can bundle your solution artefacts and see their location.

```python
solution.save()
solution.location
```

After obtaining the `/path/to/your/artefacts` you can upload it as a bundle to [CodaLab](https://codalab.org/):

```bash
cl upload /path/to/your/artefacts
```

This command will display the URL of your uploaded bundle. It assumes that you have a user account on CodaLab ([click here](https://codalab-worksheets.readthedocs.io/en/latest/features/bundles/uploading/) for more info).

After that, you simply go to our [FORM LINK](#), fill it in with all required details and paste the bundle link so we can run a full evaluation on it.

Congratulations! Your solution is now uploaded to our system and after evaluation it will show up on the [leaderboard](#).

## Adding a Submitted solution to the Repo

This step is performed manually by us (although it could be possible to automate). It looks like this:

```bash
dcai add-solution \
    --scenario miniclean/bank \
    --name MySolution \
    --paper https://arxiv.org/abs/... \
    --code https://github.com/... \
    --artefacts-url https://worksheets.codalab.org/rest/bundles/...
```

## Performing the Full Evaluation

This step is performed by GitHub Actions and is triggered after each commit.

```bash
dcai evaluate --leaderboard-output /path/to/leaderboard/dir
```
