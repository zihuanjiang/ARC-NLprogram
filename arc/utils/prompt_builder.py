# arc/utils/prompt_builder.py
import os
from typing import Optional

def load_prompt(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, 'prompts', filename)
    with open(prompt_path, 'r') as f:
        return f.read().strip()

EXECUTOR_SYSTEM = load_prompt('executor_system.txt')

pair = "Input Grid:\n{inp}\n\nOutput Grid:\n{out}"
PAIR_WITH_ABSTRACTION = "Input Grid:\n{inp}\nInput Grid Json:\n{inp_json}\nOutput Grid:\n{out}\nOutput Grid Json:\n{out_json}\n"

def grid_to_ascii(g):
    return "\n".join("".join(str(int(x)) for x in row) for row in g)

def build_executor_prompt(task,instructions: str,abstractions,include_train_input: bool = True,include_test_input: bool = True, include_nl_program: bool = True):
    
    prompt = instructions if include_nl_program else ""
    
    if include_train_input:
        for i, ex in enumerate(task.trainingExamples):
            prompt += f"\nExample {i+1}:\n"
            inp = grid_to_ascii(ex["input"])
            out = grid_to_ascii(ex["output"])
            if abstractions is not None:
                inp_json = abstractions['train'][i]['input']
                out_json = abstractions['train'][i]['output']
                prompt += PAIR_WITH_ABSTRACTION.format(inp=inp, inp_json=inp_json, out=out, out_json=out_json)
            else:
                prompt += pair.format(inp=inp, out=out)

    if include_test_input:
        header = "Now, apply the instructions to the following test input."
        prompt += header
        test_inp = grid_to_ascii(task.testExamples[0]["input"])

        test_inp_obj = None

        if abstractions is not None:
            # Safely extract the specific test object if it exists
            test_list = abstractions.get("test")
            if test_list and len(test_list) > 0:
                test_inp_obj = test_list[0]

        # Now safe to check
        if test_inp_obj:
            prompt += f"\nTest Input Grid:\n\n{test_inp}\nTest Input Grid Json:\n{test_inp_obj}"
        else:
            prompt += f"\nTest Input Grid:\n\n{test_inp}"

    grid_prompt = prompt.strip()

    return EXECUTOR_SYSTEM, grid_prompt


GENERATOR_SYSTEM_PROMPT = load_prompt('generator_system.txt')

def build_generator_prompt(task, abstractions, include_train_input = True,include_test_input=True, include_abstraction: bool = True):
    prompt = ''
    
    if include_train_input:
        for i, ex in enumerate(task.trainingExamples):
            prompt += f"\nExample {i+1}:\n"
            inp = grid_to_ascii(ex["input"])
            out = grid_to_ascii(ex["output"])
            if abstractions is not None and include_abstraction:
                inp_json = abstractions['train'][i]['input']
                out_json = abstractions['train'][i]['output']
                prompt += PAIR_WITH_ABSTRACTION.format(inp=inp, inp_json=inp_json, out=out, out_json=out_json)
            else:
                prompt += pair.format(inp=inp, out=out)

    if include_test_input:
        test_inp = grid_to_ascii(task.testExamples[0]["input"])
        prompt += f"\nTest Input Grid:\n\n{test_inp}"

    grid_prompt = prompt.strip()

    return GENERATOR_SYSTEM_PROMPT, grid_prompt


def build_executor_prompt_v2(
    task,
    instructions: str,
    abstractions,
    include_train_input: bool = True,
    include_test_input: bool = True,
    include_nl_program: bool = True,
    test_train_accuracy: bool = False,
    train_example_index: Optional[int] = None
):
    """
    Build executor prompt v2 that supports generating outputs for train inputs.
    
    Args:
        task: ARCTask object
        instructions: Natural language instructions
        abstractions: Optional abstractions dict
        include_train_input: Whether to include training examples in prompt (for reference)
        include_test_input: Whether to include test input
        include_nl_program: Whether to include the instructions
        test_train_accuracy: If True, generate output for a training example (don't show correct answer)
        train_example_index: Which training example to generate output for (if test_train_accuracy is True)
    
    Returns:
        tuple: (system_prompt, grid_prompt)
    """
    prompt = instructions if include_nl_program else ""
    
    # Include training examples for reference
    # When testing train accuracy, skip the example we're testing (we'll show it separately at the end)
    if include_train_input:
        for i, ex in enumerate(task.trainingExamples):
            # Skip the example we're testing - we'll show it separately at the end
            if test_train_accuracy and train_example_index is not None and i == train_example_index:
                continue
                
            prompt += f"\nExample {i+1}:\n"
            inp = grid_to_ascii(ex["input"])
            out = grid_to_ascii(ex["output"])
            
            # Show full example with output (for reference)
            if abstractions is not None:
                inp_json = abstractions['train'][i]['input']
                out_json = abstractions['train'][i]['output']
                prompt += PAIR_WITH_ABSTRACTION.format(inp=inp, inp_json=inp_json, out=out, out_json=out_json)
            else:
                prompt += pair.format(inp=inp, out=out)
    
    # Include test input if requested
    if include_test_input and not test_train_accuracy:
        header = "Now, apply the instructions to the following test input."
        prompt += f"\n{header}"
        test_inp = grid_to_ascii(task.testExamples[0]["input"])
        
        test_inp_obj = None
        if abstractions is not None:
            test_list = abstractions.get("test")
            if test_list and len(test_list) > 0:
                test_inp_obj = test_list[0]
        
        if test_inp_obj:
            prompt += f"\nTest Input Grid:\n\n{test_inp}\nTest Input Grid Json:\n{test_inp_obj}"
        else:
            prompt += f"\nTest Input Grid:\n\n{test_inp}"
    
    # If testing train accuracy, add the specific train input to solve
    if test_train_accuracy and train_example_index is not None:
        header = "Now, apply the instructions to the following test input."
        prompt += f"\n{header}"
        train_inp = grid_to_ascii(task.trainingExamples[train_example_index]["input"])
        
        train_inp_obj = None
        if abstractions is not None:
            train_list = abstractions.get("train")
            if train_list and len(train_list) > train_example_index:
                train_inp_obj = train_list[train_example_index]['input']
        
        if train_inp_obj:
            prompt += f"\nTraining Input Grid:\n\n{train_inp}\nTraining Input Grid Json:\n{train_inp_obj}"
        else:
            prompt += f"\nTraining Input Grid:\n\n{train_inp}"
    
    grid_prompt = prompt.strip()
    
    return EXECUTOR_SYSTEM, grid_prompt