import os

import pandas as pd
from tabulate import tabulate

import dcbench
from dcbench.common.artefact import ArtefactContainerClass


def get_artefact_table(task: ArtefactContainerClass):
    df = pd.DataFrame(
        [
            {
                "name": f"``{name}``",
                "type": f":class:`dcbench.{spec.artefact_type.__name__}`",
                "description": spec.description,
            }
            for name, spec in task.artefact_specs.items()
        ]
    ).set_index(keys="name")

    return tabulate(df, headers="keys", tablefmt="rst")


sections = ["🎯 Tasks\n========="]
for task in dcbench.tasks:
    template = open(os.path.join("source/task_templates", f"{task.task_id}.rst")).read()
    section = template.format(
        summary=task.summary,
        num_problems=len(task.instances),
        artefact_table=get_artefact_table(task),
        attributes_table="",
    )

    sections.append(section)

open("source/tasks.rst", "w").write("\n\n".join(sections))
