import os

import pandas as pd
from tabulate import tabulate

import dcbench
from dcbench.common.artefact import ArtefactContainerClass


def get_rst_class_ref(klass: type):
    return (f":class:`dcbench.{klass.__name__}`",)


def get_artefact_table(task: ArtefactContainerClass):
    df = pd.DataFrame(
        [
            {
                "name": f"``{name}``",
                "type": get_rst_class_ref(spec.artefact_type),
                "description": spec.description,
            }
            for name, spec in task.artefact_specs.items()
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
        task_name=task.full_name,
        summary=task.summary,
        num_problems=len(task.instances),
        problem_class=get_rst_class_ref(task),
        problem_artefact_table=get_artefact_table(task),
        solution_class=get_rst_class_ref(task.solution_class),
        solution_artefact_table=get_artefact_table(task),
        storage_link=task.remote_instances_url,
        longer_description=longer_description,
    )

    sections.append(section)

open("source/tasks.rst", "w").write("\n\n".join(sections))
