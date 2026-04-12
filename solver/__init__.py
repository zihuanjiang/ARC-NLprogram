"""
Interpreter-executor solver for ARC tasks.

This solver uses a three-component pipeline:
  1. Interpreter — parses each instruction line into an atomic action
  2. Executor   — executes the atomic action (updating grid/state)
  3. PC Updater — advances the program counter to the next instruction

The architecture emulates a small virtual machine that runs natural-language
programs over ARC grids.
"""
