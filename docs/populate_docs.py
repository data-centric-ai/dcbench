import os

import pandas as pd
from tabulate import tabulate

import dcbench
from dcbench.common.artifact import ArtifactContainerClass

BUCKET_BROWSER_URL = "https://console.cloud.google.com/storage/browser/dcbench"


def get_rst_class_ref(klass: type):
    return f":class:`dcbench.{klass.__name__}`"


def get_link(text: str, url: str):
    return f"`{text} <{url}>`_"


def get_artifact_table(task: ArtifactContainerClass):
    df = pd.DataFrame(
        [
            {
                "name": f"``{name}``",
                "type": get_rst_class_ref(spec.artifact_type),
                "description": spec.description,
            }
            for name, spec in task.artifact_specs.items()
        ]
    ).set_index(keys="name")

    return tabulate(df, headers="keys", tablefmt="rst")


sections = ["🎯 Tasks\n========="]
template = open("source/task_template.rst").read()
for task in dcbench.tasks:
    longer_description = open(
        os.path.join("source/task_descriptions", f"{task.task_id}.rst")
    ).read()
    section = template.format(
        full_name=task.full_name,
        summary=task.summary,
        num_problems=len(task.instances),
        task_id=task.task_id,
        problem_class=get_rst_class_ref(task),
        problem_artifact_table=get_artifact_table(task),
        solution_class=get_rst_class_ref(task.solution_class),
        solution_artifact_table=get_artifact_table(task.solution_class),
        storage_url=os.path.join(BUCKET_BROWSER_URL, task.task_id),
        longer_description=longer_description,
    )

    sections.append(section)

open("source/tasks.rst", "w").write("\n\n".join(sections))
