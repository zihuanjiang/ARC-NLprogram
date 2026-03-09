"""
ARC solvers package.
Contains different solver implementations for ARC tasks.
"""

from .base import Solver
from .typed_sequential_solver import TypedSequentialSolver
from .monolithic_solver import MonolithicSolver

__all__ = [
    'Solver',
    'TypedSequentialSolver',
    'MonolithicSolver',
]
