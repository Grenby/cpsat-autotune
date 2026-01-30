from typing import TypeVar, Protocol, runtime_checkable
from enum import Enum

M = TypeVar('M')

class SolutionStatus(Enum):
    FEASIBLE = 0
    INFEASIBLE = 1
    OPTIMAL = 2
    NOT_FOUND = 3

@runtime_checkable
class ModelSolver(Protocol[M]):

    @property
    def seed(self) -> int:
        pass
    
    @seed.setter
    def seed(self,_seed:int):
        pass

    def solve(self, model : M)->SolutionStatus:
        pass
    
    @property
    def best_objective(self):
        pass
    
    @property
    def best_objective(self):
        pass
    
    @property
    def best_bound(self):
        pass

    @property
    def max_time_seconds(self):
        pass

    @max_time_seconds.setter
    def max_time_seconds(self, time_seconds : int):
        pass

@runtime_checkable
class GapModelSolver(Protocol[M]):
    @property
    def relative_gap_limit(self):
        pass

    @property
    def absolute_gap_limit(self):
        pass

    @relative_gap_limit.setter
    def relative_gap_limit(self, _value : int| float):
        pass

    @absolute_gap_limit.setter
    def absolute_gap_limit(self, _value : int| float):
        pass

@runtime_checkable
class GapIntervalModelSolver(Protocol[M]):
    @property
    def gap_integral(self):
        pass

@runtime_checkable
class SolverFactory(Protocol[M]):
    def prepare_solver(self, params: dict[str, float | int | bool | list | tuple]) -> ModelSolver[M]:
        pass
