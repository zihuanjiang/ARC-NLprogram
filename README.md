<div align="center">

# ARC-NLprogram

Solving ARC (Abstraction and Reasoning Corpus) tasks with large language models and natural-language programs.

</div>

---

## Overview

This repository is the **open-source branch** of an ongoing research project that explores different strategies for using LLMs to solve ARC tasks. Each task presents a small number of input–output grid examples from which a transformation rule must be inferred and applied to a new test input.

Two independent solver implementations are provided here. They share the same goal but take fundamentally different approaches:

| | v0 — Modular Pipeline | v1 — Interpreter-Executor |
|-|----------------------|--------------------------|
| **Approach** | Decompose solving into abstraction, generation, execution, and evaluation stages | Emulate a virtual machine that iteratively steps through a natural-language program |
| **Configuration** | Hydra YAML | CLI arguments |
| **Key components** | Abstractor, Program Generator, Program Executor, Judge, Evaluator | Interpreter, Executor, PC Updater |

## Repository Structure

```
ARC-NLprogram/
├── README.md               ← you are here
├── v0/                     ← Modular pipeline solver (Solver Group A)
│   ├── README.md
│   ├── main.py             ← Hydra CLI entrypoint
│   ├── requirements.txt
│   ├── config/             ← Hydra YAML configs
│   └── arc/                ← Components, evaluator, solvers, utilities
└── v1/                     ← Interpreter-executor solver (Solver Group B)
    ├── README.md
    ├── run.py              ← CLI entrypoint
    ├── requirements.txt
    ├── solver/             ← Interpreter, executor, PC updater, runner
    └── arc/                ← Data loader, LLM provider, utilities
```

Each subfolder is self-contained with its own dependencies, entrypoint, and documentation. Refer to the individual README files for setup and usage instructions:

- [`v0/README.md`](v0/README.md)
- [`v1/README.md`](v1/README.md)

## What Is Open-Sourced

- **Architecture and interfaces** — all component base classes, registries, and solver orchestration logic
- **Pipeline implementations** — end-to-end flow from task loading through solving to evaluation
- **Evaluation metrics** — pointwise accuracy, success rate, weighted F1, ARGA weighted error
- **Control-flow logic** — program-counter management, structure parsing
- **Utilities** — answer parsing, prompt building, grid plotting, solver helpers
- **Configuration system** — Hydra YAML structure (v0) and CLI arguments (v1)

## What Is Not Open-Sourced

The following are replaced with clearly marked redacted placeholders (`[REDACTED — closed-source prompt]` or stubs):

- **LLM prompts** — all system prompts and prompt templates used for abstraction, generation, execution, judging, interpretation, and PC-updating
- **Pydantic structured-output schemas** — model definitions used for LLM response parsing
- **Model configurations** — specific model names, provider endpoints, and parameter settings
- **Task data** — ARC dataset files are not bundled (download from [ARC Prize](https://arcprize.org/))

## Usage

### v0 — Modular Pipeline

```bash
cd v0
pip install -r requirements.txt

export OPENROUTER_API_KEY=<your-key>
# Add model entries to arc/components/llm/config.py

python main.py experiment=sample solver_type=monolithic experiment_name=my_run
```

### v1 — Interpreter-Executor

```bash
cd v1
pip install -r requirements.txt

export OPENROUTER_API_KEY=<your-key>
# Add model entries to arc/components/llm/config.py

python run.py \
    --task_id 2204b7a8 \
    --data_folder /path/to/arc-data \
    --instruction program.txt \
    --max_steps 500 \
    --output result.json
```

## News

- **2025-03** — Initial open-source release with v0 (modular pipeline) and v1 (interpreter-executor) solvers.

## License

This project is released for academic and research use. See the LICENSE file for details.

## Acknowledgements

This project builds on the [ARC Prize](https://arcprize.org/) benchmark introduced by François Chollet.
