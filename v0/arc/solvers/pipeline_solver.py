from arc.solvers.registry import AbstractorRegistry, ProgramGeneratorRegistry, ProgramExecutorRegistry, JudgeRegistry
from arc.evaluator.pointwise_acc import Pointwise_acc
from arc.evaluator.success_rate import Success_rate
from arc.evaluator.weighted_f1 import Weighted_f1
from arc.evaluator.foreground_acc import Foreground_acc
from arc.components.judge.llm_judge import LLMJudge
from arc.components.judge.python_proxy import PythonProxyJudge
from arc.data.ARCTask import ARCTask

from arc.solvers.base import Solver

class FullPipelineSolver(Solver):
    def __init__(self, abstraction_config, program_gen_config, executor_config, judge_config=None):
        # Instantiate Abstractor if config is provided
        self.abstractor = None
        if abstraction_config:
            abs_impl_name = abstraction_config.pop("implementation", "v1")
            AbstractorClass = AbstractorRegistry.get_implementation(abs_impl_name)
            self.abstractor = AbstractorClass(**abstraction_config)

        # Instantiate Program Generator
        gen_impl_name = program_gen_config.pop("implementation", "llm")
        GeneratorClass = ProgramGeneratorRegistry.get_implementation(gen_impl_name)
        self.program_generator = GeneratorClass(**program_gen_config)

        # Instantiate Program Executor
        exec_impl_name = executor_config.pop("implementation", "llm")
        ExecutorClass = ProgramExecutorRegistry.get_implementation(exec_impl_name)
        self.program_executor = ExecutorClass(**executor_config)

        # Instantiate Judges
        self.judges = []
        if judge_config:
            # Normalize to list
            if isinstance(judge_config, dict):
                judge_configs = [judge_config]
            else:
                judge_configs = judge_config
            
            for j_conf in judge_configs:
                j_conf = j_conf.copy()
                impl_name = j_conf.pop("implementation", "llm")
                JudgeClass = JudgeRegistry.get_implementation(impl_name)
                self.judges.append(JudgeClass(**j_conf))

        self.metrics = [Pointwise_acc(), Success_rate(), Weighted_f1(), Foreground_acc()]

    def solve(self, task: ARCTask, experiment_name: str = "default_experiment") -> dict:

        print(f"\n--- Solving Task {task.task_id} ---")
        
        total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        def aggregate_usage(usage_list):
            if not isinstance(usage_list, list):
                usage_list = [usage_list]
            for u in usage_list:
                if u:
                    total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += u.get("completion_tokens", 0)
                    total_usage["total_tokens"] += u.get("total_tokens", 0)

        # Generate object abstraction only if needed
        abstractions = None
        if self.abstractor and (self.program_generator.include_abstraction or self.program_executor.include_abstraction):
            abstractions, abstractor_usage = self.abstractor.abstract(task)
            aggregate_usage(abstractor_usage)
        else:
            print("Skipping abstraction generation (not needed for this experiment)")

        # Generate the natural language program 
        generation_output = self.program_generator.generate(
            task=task,
            abstractions=abstractions
        )
        aggregate_usage(generation_output.get("usage"))

        instructions = generation_output.get('instructions', None)

        if not instructions:
            print("Generation stage failed to produce instructions. Aborting solve.")
            return {
                "status": "failed_at_generation",
                "generation_output": generation_output,
                "predicted_grid": None,
                "total_usage": total_usage
            }
        
        print(f"Generated Instructions:\n{instructions}\n")

        # Judge the instructions
        judge_outputs = []
        if self.judges:
            for judge in self.judges:
                output = judge.judge(task, instructions)
                output['judge_name'] = judge.__class__.__name__
                judge_outputs.append(output)
                aggregate_usage(output.get("usage"))
                print(f"Judge ({output['judge_name']}) Verdict: {output.get('verdict')} (Confidence: {output.get('confidence')})")

        # Execute the program
        execution_output = self.program_executor.execute(
            task=task,
            instructions=instructions,
            abstractions=abstractions
        )
        aggregate_usage(execution_output.get("usage"))

        # Evaluate the result
        predicted_grids = [execution_output['predicted_grid']]
        
        # Initialize list of dicts for each test example
        evaluation_results = [{} for _ in range(len(task.testExamples))]
        
        for metric in self.metrics:
            scores = metric.run(task, predicted_grids)
            for i, score in enumerate(scores):
                evaluation_results[i][metric.name] = score
                
        task_evaluation = evaluation_results
        print(f"Task evaluation result: {task_evaluation}")
        print(f"Total Tokens: {total_usage['total_tokens']}")

        final_log = {
            "status": "success",
            "task_id": task.task_id,
            "grid_abstractions": abstractions,
            "predicted_grid": execution_output.get("predicted_grid"),
            "generation_output": generation_output,
            "judge_output": judge_outputs,
            "execution_output": execution_output,
            "task_evaluation": task_evaluation,
            "total_usage": total_usage,
            "debug": {
            "full_generation_output": generation_output,
            "full_execution_output": execution_output,
            "abstractions": abstractions
            }
        }

        print(f"--- Finished Task {task.task_id} ---")
        
        return final_log
