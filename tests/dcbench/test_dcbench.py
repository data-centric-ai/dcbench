import pandas as pd

import dcbench
from dcbench.common.table import Table
from dcbench.common.task import Task


def test_tasks():
    assert isinstance(dcbench.tasks, Table)
    assert len(dcbench.tasks) == 3


def test_tasks_html():
    dcbench.tasks._repr_html_()


def test_tasks_df():
    df = dcbench.tasks.df
    assert isinstance(df, pd.DataFrame)


def test_get_tasks():
    for task_id in dcbench.tasks:
        out = dcbench.tasks[task_id]
        assert isinstance(out, Task)
        assert task_id == out.id
