from tqdm import tqdm
from typing import Dict, Sequence, Optional, Callable, TYPE_CHECKING

from .table import RowUnion, Table

if TYPE_CHECKING:
    from .problem import Problem
    from .result import Result
    from .solution import Solution
else:
    class Problem:
        pass
    class Solution:
        pass


class Trial(Table):

    def __init__(self, problems: Optional[Sequence[Problem]] = None,
                 solver: Optional[Callable[[Problem], Solution]] = None):

        self.problems = problems or []
        self.solver = solver
        self.solutions: Dict[str, Solution] = {}
        self.results: Dict[str, Result] = {}
        super().__init__([])

    def evaluate(self, repeat: int = 1, quiet: bool = False) -> "Trial":
        assert(repeat >= 1)
        assert(self.solver is not None)

        problems = self.problems if quiet else tqdm(self.problems, desc="Problems")
        for problem in problems:
            repetitions = range(repeat) if quiet or repeat == 1 else tqdm(range(repeat), desc="Repetitions")
            for _ in repetitions:
                solution = self.solver(problem)
                result = problem.evaluate(solution)
                self.solutions[solution.id] = solution
                self.results[solution.id] = result
                self._add_row(RowUnion(id=solution.id, elements=[problem, solution, result]))
        return self

    def save(self) -> None:
        raise NotImplementedError()
