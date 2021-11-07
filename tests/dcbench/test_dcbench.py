import dcbench


def test_tasks():
    tasks = dcbench.tasks
    assert len(tasks) > 0
    assert dcbench.SliceDiscoveryProblem in tasks
