import os
import json


class ARCTask:
    def __init__(self, folder, set):
        self.set = set
        self.challenges_file = os.path.join(folder, f"arc-agi_{self.set}_challenges.json")
        self.solutions_file = os.path.join(folder, f"arc-agi_{self.set}_solutions.json")
        self.task_id = None
        self.trainingExamples = []
        self.testExamples = []

    def load(self, task_id):
        self.task_id = task_id

        with open(self.challenges_file, "r", encoding="utf-8") as f:
            challenges_data = json.load(f)

        with open(self.solutions_file, "r", encoding="utf-8") as f:
            solutions_data = json.load(f)

        task_data = challenges_data[task_id]
        task_solutions = solutions_data[task_id]

        self.trainingExamples = task_data["train"]

        self.testExamples = []
        for i, test_input_case in enumerate(task_data["test"]):
            self.testExamples.append({
                "input": test_input_case["input"],
                "output": task_solutions[i]
            })

        return self
