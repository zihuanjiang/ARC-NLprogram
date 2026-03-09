import matplotlib
matplotlib.use('Agg')

import os
import json
import concurrent.futures
import threading

from arc.data.ARCTask import ARCTask
from arc.utils.plotting import plot_solution_comparison, plot_task

print_lock = threading.Lock()


def process_task(task_id, solver, exp_name, exp_dir, data_folder_path, dataset_set, solver_config):
    with print_lock:
        print(f"\nProcessing Task: {task_id}")

    task_dir = os.path.join(exp_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)

    try:
        task = ARCTask(folder=data_folder_path, set=dataset_set).load(task_id)
    except Exception as e:
        with print_lock:
            print(f"Error loading task {task_id}: {e}")
        return

    try:
        plot_task(task, save_image=True, save_path=task_dir, show_image=False)
    except Exception as e:
        with print_lock:
            print(f"Error plotting task: {e}")

    try:
        solution_log = solver.solve(task, experiment_name=exp_name)

        log_dir = os.path.join(task_dir, 'solution_log')
        os.makedirs(log_dir, exist_ok=True)

        def save_json(data, filename, target_dir):
            path = os.path.join(target_dir, filename)
            with open(path, 'w') as f:
                json.dump(
                    data, f,
                    default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x),
                    indent=2,
                )

        save_json(solver_config, 'experiment_config.json', log_dir)

        if 'grid_abstractions' in solution_log:
            save_json(solution_log['grid_abstractions'], 'grid_abstraction.json', log_dir)

        if 'generation_output' in solution_log:
            save_json(solution_log['generation_output'], 'generation_output.json', log_dir)

        test_dir = os.path.join(log_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)

        execution_output = solution_log.get('execution_output', {})
        test_execution_output = execution_output.get('test_execution_output', execution_output)

        if test_execution_output:
            save_json(test_execution_output, 'execution_output.json', test_dir)

        test_summary = {
            "status": solution_log.get("status"),
            "task_id": solution_log.get("task_id"),
            "task_evaluation": solution_log.get("task_evaluation"),
            "predicted_grid": solution_log.get("predicted_grid"),
            "total_usage": solution_log.get("total_usage"),
        }
        save_json(test_summary, 'prediction_summary.json', test_dir)

        train_evaluation = solution_log.get("train_evaluation")
        train_execution_outputs = execution_output.get('train_execution_outputs', [])
        predicted_train_grids = execution_output.get('predicted_train_grids', [])

        if train_evaluation is not None and len(train_evaluation) > 0:
            for train_idx in range(len(train_evaluation)):
                train_dir = os.path.join(log_dir, f'train_{train_idx}')
                os.makedirs(train_dir, exist_ok=True)

                if train_idx < len(train_execution_outputs):
                    save_json(train_execution_outputs[train_idx], 'execution_output.json', train_dir)

                train_summary = {
                    "status": solution_log.get("status"),
                    "task_id": solution_log.get("task_id"),
                    "task_evaluation": [train_evaluation[train_idx]],
                    "predicted_grid": predicted_train_grids[train_idx] if train_idx < len(predicted_train_grids) else None,
                    "total_usage": solution_log.get("total_usage"),
                }
                save_json(train_summary, 'prediction_summary.json', train_dir)

        if solution_log and 'predicted_grid' in solution_log and solution_log['predicted_grid'] is not None:
            target_grid = task.testExamples[0]['output']
            test_input_grid = task.testExamples[0]['input']
            prediction = solution_log['predicted_grid']

            plot_solution_comparison(
                task.task_id, test_input_grid, prediction, target_grid,
                save_image=True, save_path=test_dir, show_image=False,
            )

        predicted_train_grids = execution_output.get('predicted_train_grids', [])
        if predicted_train_grids:
            for train_idx, predicted_grid in enumerate(predicted_train_grids):
                train_dir = os.path.join(log_dir, f'train_{train_idx}')
                train_ex = task.trainingExamples[train_idx]
                plot_solution_comparison(
                    f"{task.task_id}_train_{train_idx}",
                    train_ex['input'], predicted_grid, train_ex['output'],
                    save_image=True, save_path=train_dir, show_image=False,
                )

    except Exception as e:
        with print_lock:
            print(f"Error solving task {task_id}: {e}")
        import traceback
        traceback.print_exc()


def run_experiment(cfg):
    from omegaconf import OmegaConf
    from arc.solvers.typed_sequential_solver import TypedSequentialSolver
    from arc.solvers.monolithic_solver import MonolithicSolver

    if cfg.get('solver_type') is None:
        raise ValueError(
            "solver_type must be specified. "
            "Example: solver_type=monolithic or solver_type=typed_sequential"
        )

    solver_type = str(cfg.get('solver_type')).lower()

    experiment_config = cfg.get('experiment', {})
    task_ids = OmegaConf.to_container(experiment_config.get('tasks', []), resolve=True) if hasattr(experiment_config, 'tasks') else experiment_config.get('tasks', [])
    if not task_ids:
        print("No tasks specified in experiment config")
        return

    exp_name = cfg.get('experiment_name') or getattr(experiment_config, '_name_', None) or "experiment"

    solver_config = cfg.get('solver', {})
    solver_config_dict = OmegaConf.to_container(solver_config, resolve=True) if hasattr(solver_config, '__class__') and 'OmegaConf' in str(type(solver_config)) else solver_config

    print(f"\n{'='*50}")
    print(f"Starting Experiment: {exp_name}")
    print(f"Solver Type: {solver_type}")
    print(f"Tasks: {len(task_ids)}")
    print(f"Parallel Workers: {cfg.get('workers', 1)}")
    print(f"{'='*50}")

    exp_dir = os.path.join('experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    try:
        if solver_type == "monolithic":
            solver = MonolithicSolver(config_updates=solver_config_dict)
        elif solver_type == "typed_sequential":
            solver = TypedSequentialSolver(config_updates=solver_config_dict)
        else:
            raise ValueError(f"Unknown solver_type: {solver_type}. Supported: 'monolithic', 'typed_sequential'")
    except Exception as e:
        print(f"Error initializing {solver_type} solver: {e}")
        import traceback
        traceback.print_exc()
        return

    dataset_config = cfg.get('dataset', {})
    dataset_folder = dataset_config.get('folder')
    dataset_set = dataset_config.get('set')

    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.get('workers', 1)) as executor:
        futures = [
            executor.submit(process_task, task_id, solver, exp_name, exp_dir, dataset_folder, dataset_set, solver_config_dict)
            for task_id in task_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during task execution: {e}")
