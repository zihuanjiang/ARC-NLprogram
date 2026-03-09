# v1 — Interpreter-Executor ARC Solver

An experimental ARC solver that emulates a small virtual machine to execute natural-language programs over grids. Instead of a single monolithic LLM call, this solver decomposes execution into three cooperating LLM-powered components that iteratively step through the program.

## Architecture

```
NL Program (multi-line instruction text)
  │
  ├─► Interpreter   — parse current line → atomic action
  │
  ├─► Executor      — execute action
  │
  └─► PC Updater    — advance program counter to next line
         │
         └─── loop until RETURN or max_steps
```

| Component | Role |
|-----------|------|
| **Interpreter** | Reads the current instruction line and emits a single atomic action |
| **Executor** | Carries out the action and returns a result |
| **PC Updater** | Decides which line to execute next based on the action result and program structure |

## Directory Structure

```
v1/
├── run.py                       # CLI entrypoint
├── requirements.txt
├── solver/
│   ├── __init__.py
│   ├── interpreter.py           # Instruction → atomic action
│   ├── executor.py              # Action dispatch (Python + LLM)
│   ├── pc_updater.py            # Program-counter logic
│   └── runner.py                # Orchestration loop
└── arc/
    ├── data/ARCTask.py          # Task data loader
    ├── components/llm/          # LLM provider + model config
    └── utils/                   # Plotting, parsing, prompt building
```

## Setup

```bash
cd v1
pip install -r requirements.txt
```

### Prerequisites

1. **ARC task data** — download the [ARC Prize 2024](https://arcprize.org/) dataset.
2. **API key** — set the `OPENROUTER_API_KEY` environment variable.
3. **Model configurations** — add your model entries to `arc/components/llm/config.py`.
4. **NL program** — prepare a text file containing the natural-language program to execute.

## Usage

```bash
python run.py \
    --task_id 2204b7a8 \
    --data_folder /path/to/arc-prize-2024 \
    --instruction program.txt \
    --model_key my-model \
    --max_steps 500 \
    --output result.json
```

### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--task_id` | Yes | ARC task ID to solve |
| `--data_folder` | Yes | Path to the ARC data directory |
| `--instruction` | Yes | Path to a `.txt` file with the NL program |
| `--model_key` | No | Key from `MODEL_CONFIGURATIONS` (defaults to first entry) |
| `--dataset_set` | No | Dataset split — `training` (default) or `evaluation` |
| `--max_steps` | No | Safety limit on execution steps (default: 500) |
| `--time_sleep` | No | Seconds to sleep between LLM calls (default: 1) |
| `--output` | No | Path to write the output grid as JSON |

## What Is Open-Sourced

- Three-component solver architecture (interpreter, executor, PC updater)
- Program-counter control-flow logic (structure parsing, brace matching)
- Runner / orchestration loop
- Task data loading and visualisation utilities

## What Is Not Open-Sourced

- **Prompts** — all LLM prompt contents are replaced with redacted placeholders
- **Executor logic** — action dispatch and execution internals are redacted
- **Model configurations** — specific model names and provider settings are removed
- **Task data** — ARC dataset files are not included (download separately)

## License

See the top-level repository LICENSE file.
