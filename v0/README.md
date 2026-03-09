# v0 — Modular Pipeline ARC Solver

A modular, component-based solver for the [ARC (Abstraction and Reasoning Corpus)](https://arcprize.org/) challenge. This implementation decomposes the solving process into independently configurable stages connected through a pipeline architecture.

## Architecture

The solver follows a multi-stage pipeline:

```
ARC Task
  │
  ├─► Abstraction  (optional — grid → structured object representation)
  │
  ├─► Generation   (training pairs → natural-language program)
  │
  ├─► Execution    (NL program + test input → predicted output grid)
  │
  └─► Evaluation   (predicted grid vs. ground truth → metrics)
```

Two top-level solver implementations are provided:

| Solver | Description |
|--------|-------------|
| `MonolithicSolver` | Single-pass pipeline: abstraction → generation → execution |
| `TypedSequentialSolver` | Sequential step-by-step execution with instruction categorisation |

## Directory Structure

```
v0/
├── main.py                      # Entrypoint (Hydra CLI)
├── requirements.txt
├── config/                      # Hydra YAML configurations
│   ├── config.yaml
│   ├── solver/
│   └── experiment/
└── arc/
    ├── cli.py                   # Hydra wiring
    ├── run.py                   # Experiment runner (parallel task execution)
    ├── data/ARCTask.py          # Task data loader
    ├── components/
    │   ├── abstraction/         # Grid abstraction (LLM / heuristic / dynamic)
    │   ├── judge/               # Instruction verification (LLM / Python proxy)
    │   ├── llm/                 # LLM provider + model config
    │   ├── program_executor/    # NL program → output grid
    │   └── program_generator/   # Training pairs → NL program
    ├── evaluator/               # Metrics (pointwise accuracy, F1, etc.)
    ├── solvers/                 # Top-level solver orchestrators
    ├── slm_executor/            # Shared utilities
    └── utils/                   # Prompt building, plotting, parsing
```

## Setup

```bash
cd v0
pip install -r requirements.txt
```

### Prerequisites

1. **ARC task data** — download the [ARC Prize 2024](https://arcprize.org/) dataset and place it under a `data/` directory (or point to it via config).
2. **API key** — set the `OPENROUTER_API_KEY` environment variable.
3. **Model configurations** — add your model entries to `arc/components/llm/config.py`.

## Usage

### Quick start

```bash
# Run a single task with the MonolithicSolver
python main.py experiment=sample solver_type=monolithic experiment_name=my_run

# Run with TypedSequentialSolver
python main.py experiment=sample solver_type=typed_sequential experiment_name=my_run
```

### CLI overrides (Hydra)

```bash
python main.py \
    experiment=sample \
    solver_type=monolithic \
    experiment_name=my_experiment \
    solver.abstraction_config.model_key=my-model \
    workers=4
```

Results are saved under `experiments/<experiment_name>/`.

## What Is Open-Sourced

- Full pipeline architecture and component interfaces
- Solver orchestration logic
- Evaluation metrics (pointwise accuracy, success rate, weighted F1, ARGA weighted error)
- Task data loading and visualisation utilities
- Configuration system (Hydra YAML)

## What Is Not Open-Sourced

- **Prompts** — all LLM prompt contents are replaced with redacted placeholders
- **Pydantic schemas** — structured output model definitions are replaced with stubs
- **Model configurations** — specific model names and provider settings are removed
- **Task data** — ARC dataset files are not included (download separately)

## License

See the top-level repository LICENSE file.
