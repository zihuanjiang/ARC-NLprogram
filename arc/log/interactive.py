"""
Interactive shell for inspecting and editing an execution log.

Launch with::

    python -m arc.log.interactive <log_file.json>

Commands (type ``help`` inside the shell):
    list                      — list all recorded steps
    show <N>                  — show details for step N
    grid <N>                  — print the grid after step N
    vars <N>                  — print local variables after step N
    pc <N>                    — print program with star after step N
    set_var <N> <name> <val>  — edit a local variable at step N
    set_pc <N> <line_idx>     — move the star to a different line at step N
    set_instruction <file>    — replace the NL program from a text file
    truncate <N>              — remove all steps after N
    save [path]               — save the (edited) log
    quit                      — exit
"""
import cmd
import copy
import json
import os
import sys
from typing import Optional

from arc.log.step_logger import ExecutionLog


class InteractiveShell(cmd.Cmd):
    intro = (
        "ARC Execution Log — Interactive Shell\n"
        "Type 'help' for available commands.\n"
    )
    prompt = "(arc-log) "

    def __init__(self, log: ExecutionLog, log_path: Optional[str] = None):
        super().__init__()
        self.log = log
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_step(self, arg: str):
        try:
            n = int(arg)
        except (ValueError, TypeError):
            print(f"Invalid step number: {arg}")
            return None
        for s in self.log.steps:
            if s.step_number == n:
                return s
        print(f"Step {n} not found (range: 0..{len(self.log.steps) - 1})")
        return None

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def do_list(self, _arg: str) -> None:
        """List all recorded steps."""
        if not self.log.steps:
            print("(no steps recorded)")
            return
        print(f"{'Step':>5}  {'Action':<30}  {'PC Action':<18}  Instruction")
        print("-" * 90)
        for s in self.log.steps:
            action = s.interpreter_action[:28] if s.interpreter_action else ""
            instr = s.current_instruction[:40] if s.current_instruction else ""
            print(f"{s.step_number:>5}  {action:<30}  {s.pc_action:<18}  {instr}")

    def do_show(self, arg: str) -> None:
        """Show full details for step N.  Usage: show <N>"""
        s = self._get_step(arg)
        if s is None:
            return
        print(f"\n{'='*60}")
        print(f"  Step {s.step_number}")
        print(f"{'='*60}")
        print(f"\nCurrent instruction:\n  {s.current_instruction}")
        print(f"\nInterpreter action:\n  {s.interpreter_action}")
        print(f"\nInterpreter thought:\n  {s.interpreter_thought}")
        print(f"\nExecutor result:")
        for k, v in s.executor_result.items():
            print(f"  {k}: {v}")
        print(f"\nPC action: {s.pc_action}")
        print(f"PC thought: {s.pc_thought}")
        print(f"\nNext instruction: {s.next_instruction}")
        print(f"\nLocal variables (post):")
        lvars = s.post_state.get("local_variables", {})
        for k, v in lvars.items():
            print(f"  {k} = {v!r}")
        print()

    def do_grid(self, arg: str) -> None:
        """Print the grid after step N.  Usage: grid <N>"""
        s = self._get_step(arg)
        if s is None:
            return
        grid = s.post_memory.get("grid")
        if grid is None:
            print("No grid in memory.")
            return
        for row in grid:
            print("".join(str(c) for c in row))
        print()

    def do_vars(self, arg: str) -> None:
        """Print local variables after step N.  Usage: vars <N>"""
        s = self._get_step(arg)
        if s is None:
            return
        lvars = s.post_state.get("local_variables", {})
        if not lvars:
            print("(empty)")
        for k, v in lvars.items():
            print(f"  {k} = {v!r}")
        print()

    def do_pc(self, arg: str) -> None:
        """Print program with star after step N.  Usage: pc <N>"""
        s = self._get_step(arg)
        if s is None:
            return
        print(s.post_pc_instruction)
        print()

    def do_set_var(self, arg: str) -> None:
        """Edit a local variable at step N.  Usage: set_var <N> <name> <value>

        The value is parsed as JSON (use quotes for strings).
        All steps after N are removed."""
        parts = arg.split(None, 2)
        if len(parts) < 3:
            print("Usage: set_var <step> <name> <json_value>")
            return
        s = self._get_step(parts[0])
        if s is None:
            return
        name = parts[1]
        try:
            value = json.loads(parts[2])
        except json.JSONDecodeError:
            value = parts[2]
        lvars = copy.deepcopy(s.post_state.get("local_variables", {}))
        lvars[name] = value
        self.log.edit_step(s.step_number, local_variables=lvars)
        print(f"Set {name} = {value!r} at step {s.step_number}; "
              f"truncated to {len(self.log.steps)} steps.")

    def do_set_pc(self, arg: str) -> None:
        """Move the star to a different line at step N.
        Usage: set_pc <N> <line_index>
        All steps after N are removed."""
        parts = arg.split()
        if len(parts) < 2:
            print("Usage: set_pc <step> <line_index>")
            return
        s = self._get_step(parts[0])
        if s is None:
            return
        try:
            target_line = int(parts[1])
        except ValueError:
            print("line_index must be an integer")
            return
        lines = s.post_pc_instruction.splitlines()
        stripped = []
        for ln in lines:
            stripped.append(ln[2:] if ln.startswith("* ") else ln)
        if target_line < 0 or target_line >= len(stripped):
            print(f"line_index out of range (0..{len(stripped) - 1})")
            return
        stripped[target_line] = "* " + stripped[target_line]
        new_pc = "\n".join(stripped)
        self.log.edit_step(s.step_number, pc_instruction=new_pc)
        print(f"Moved star to line {target_line} at step {s.step_number}; "
              f"truncated to {len(self.log.steps)} steps.")

    def do_set_instruction(self, arg: str) -> None:
        """Replace the NL program from a text file.
        Usage: set_instruction <file_path>"""
        path = arg.strip()
        if not path or not os.path.isfile(path):
            print(f"File not found: {path}")
            return
        with open(path, "r") as f:
            new_instr = f.read().strip()
        self.log.replace_instruction(new_instr)
        print(f"Instruction replaced ({len(new_instr)} chars).")

    def do_truncate(self, arg: str) -> None:
        """Remove all steps after N.  Usage: truncate <N>"""
        try:
            n = int(arg)
        except (ValueError, TypeError):
            print("Usage: truncate <N>")
            return
        self.log.truncate_to(n)
        print(f"Truncated to {len(self.log.steps)} steps.")

    def do_save(self, arg: str) -> None:
        """Save the (edited) log.  Usage: save [path]"""
        path = arg.strip() or self.log_path
        if not path:
            print("No path specified and no default path set.")
            return
        self.log.save(path)
        print(f"Saved to {path}")

    def do_quit(self, _arg: str) -> bool:
        """Exit the interactive shell."""
        return True

    do_exit = do_quit
    do_EOF = do_quit


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m arc.log.interactive <log_file.json>")
        sys.exit(1)
    path = sys.argv[1]
    log = ExecutionLog.load(path)
    print(f"Loaded log: task={log.task_id}, {len(log.steps)} steps")
    shell = InteractiveShell(log, log_path=path)
    shell.cmdloop()


if __name__ == "__main__":
    main()
