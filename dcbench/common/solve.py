from typing import List

import pandas as pd
import ray

from dcbench.common.method import Method


@ray.remote
def _solve_scenario(scenario, method: Method):
    solution = scenario.solve(method)
    solution.dump()
    return solution


def solve(scenarios, methods: List[Method]) -> pd.DataFrame:
    method_refs = [ray.put(method) for method in methods]
    solution_refs = []
    for scenario in scenarios:
        for method in method_refs:
            solution_refs.append(_solve_scenario.remote(scenario, method))
    solutions = ray.get(solution_refs)
    return pd.DataFrame([solution.meta() for solution in solutions])
