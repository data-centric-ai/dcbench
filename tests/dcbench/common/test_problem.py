import pytest

import dcbench


@pytest.fixture(params=["slice_discovery"])
def problem_class(request):
    task = request.param
    if task == "slice_discovery":
        return dcbench.SliceDiscoveryProblem
    else:
        raise ValueError(f"Task '{task}' not supported.")


def test_instances(problem_class):
    instances = problem_class.instances
    assert len(instances) > 0
    assert isinstance(instances[0], problem_class)
