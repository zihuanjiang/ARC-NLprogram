"""
Interpreter-executor solver for ARC tasks.

Three-component pipeline:
  1. Interpreter — parses each instruction line into an atomic action
  2. Executor   — executes the atomic action (updating grid/state)
  3. PC Updater — advances the program counter to the next instruction
"""
from .interpreter import interpret
from .executor import execute
from .pc_updater import pc_update
from .runner import run_one_step, solve

__all__ = ["interpret", "execute", "pc_update", "run_one_step", "solve"]
