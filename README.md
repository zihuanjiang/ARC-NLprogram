# ARC-NLprogram

An ARC solver that emulates a small virtual machine to execute natural-language programs over grids. Instead of a single monolithic LLM call the solver decomposes execution into three cooperating LLM-powered components that iteratively step through the program.

## Documents

- [Defense Presentation](documents/arc_nlprogram_defense.pdf)
- [Thesis](documents/arc_nlprogram_thesis.pdf)

## Repository Layout

```
arc/                         # Python source package
├── __init__.py
├── __main__.py              # CLI entrypoint (python -m arc)
├── llm/                     # LLM provider + model config
├── data/
│   └── ARCTask.py           # Task data loader
├── solver/
│   ├── interpreter.py       # Instruction → atomic action
│   ├── executor.py          # Action dispatch (Python + LLM)
│   ├── pc_updater.py        # Program-counter logic
│   └── runner.py            # Orchestration loop
├── log/
│   ├── step_logger.py       # Structured execution logger
│   └── interactive.py       # Interactive shell for log editing
├── vis/
│   └── report.py            # PDF visualisation report
└── utils/
    ├── answer_parsing.py
    ├── plotting.py
    └── prompt_builder.py

documents/                   # Research documents
├── arc_nlprogram_defense.pdf  # Defense presentation slides
└── arc_nlprogram_thesis.pdf   # Thesis

examples/                    # Bash entrypoint scripts
└── run_2204b7a8.sh

tasks/                       # NL-program instructions per task
└── 2204b7a8/
    └── instruction.txt

data/                        # ARC dataset files
├── arc-prize-2024/
├── arc-prize-2025/
└── meta_data/
```

## Architecture

```
NL Program (multi-line instruction text)
  │
  ├─► Interpreter   — parse current line → atomic action
  │
  ├─► Executor      — execute action (Python or LLM)
  │
  └─► PC Updater    — advance program counter to next line
         │
         └─── loop until RETURN or max_steps
```

| Component       | Role                                                     |
|-----------------|----------------------------------------------------------|
| **Interpreter** | Reads the current instruction line and emits an action   |
| **Executor**    | Carries out the action and returns a result              |
| **PC Updater**  | Decides which line to execute next                       |

## Setup

```bash
pip install -r requirements.txt
```

### Prerequisites

1. **API key** — `export OPENROUTER_API_KEY="sk-..."`.
2. **Model config** — add entries to `arc/llm/config.py`.
3. **NL program** — a `.txt` file in `tasks/<task_id>/instruction.txt`.

## Usage

### Quick start

```bash
./examples/run_2204b7a8.sh
```

### Manual

```bash
python -m arc \
    --task_id 2204b7a8 \
    --data_folder data/arc-prize-2024 \
    --instruction tasks/2204b7a8/instruction.txt \
    --log_dir results/2204b7a8 \
    --report results/2204b7a8/report.pdf \
    --output results/2204b7a8/output.json \
    --max_steps 500
```

### Resume from checkpoint

```bash
python -m arc \
    --task_id 2204b7a8 \
    --data_folder data/arc-prize-2024 \
    --instruction tasks/2204b7a8/instruction.txt \
    --resume results/2204b7a8/2204b7a8_log.json \
    --log_dir results/2204b7a8 \
    --max_steps 500
```

### Interactive log editor

```bash
python -m arc.log.interactive results/2204b7a8/2204b7a8_log.json
```

Commands: `list`, `show N`, `grid N`, `vars N`, `pc N`, `set_var N name value`,
`set_pc N line_idx`, `set_instruction file`, `truncate N`, `save`, `quit`.

### PDF report

```bash
python -m arc.vis.report results/2204b7a8/2204b7a8_log.json report.pdf [max_steps]
```

## CLI Arguments

| Flag              | Required | Description                              |
|-------------------|----------|------------------------------------------|
| `--task_id`       | Yes      | ARC task ID to solve                     |
| `--data_folder`   | Yes      | Path to the ARC data directory           |
| `--instruction`   | Yes      | Path to the NL program `.txt` file       |
| `--model_key`     | No       | Key from `MODEL_CONFIGURATIONS`          |
| `--dataset_set`   | No       | `training` (default) or `evaluation`     |
| `--max_steps`     | No       | Safety limit on steps (default: 500)     |
| `--time_sleep`    | No       | Seconds between LLM calls (default: 1)  |
| `--output`        | No       | Path to write the output grid as JSON    |
| `--log_dir`       | No       | Directory to save the execution log      |
| `--resume`        | No       | Path to a log JSON to resume from        |
| `--report`        | No       | Path to save a PDF visualisation report  |

## Development History

Earlier versions of this solver are available in separate branches:

- **v0** — Modular pipeline solver (abstraction → generation → execution → evaluation)
- **v1** — Early interpreter-executor architecture (flat package layout)

## License

This project is released for academic and research use. See the LICENSE file for details.

## Acknowledgements

This project builds on the [ARC Prize](https://arcprize.org/) benchmark introduced by François Chollet.
