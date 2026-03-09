import os
from typing import Optional

from .base import ProgramGenerator
from arc.data.ARCTask import ARCTask
from arc.components.llm.provider import LLMProvider
from arc.components.llm.config import MODEL_CONFIGURATIONS
from arc.utils.prompt_builder import build_generator_prompt, GENERATOR_SYSTEM_PROMPT, grid_to_ascii
from arc.utils.plotting import base64_from_grid
from arc.solvers.registry import ProgramGeneratorRegistry

@ProgramGeneratorRegistry.register("llm")
class LLMProgramGenerator(ProgramGenerator):
    """
    An implementation of ProgramGenerator that uses an LLM to synthesize
    a natural language program from ARC task examples. This version relies
    on the API provider to separate the reasoning from the final answer.
    """
    def __init__(self, model_key: str, include_train_input: bool = True, include_test_input: bool = False, include_abstraction: bool = True, include_image: bool = False, max_tokens: Optional[int] = None, temperature: Optional[float] = None, max_attempts: int = 3):
        """
        Initializes the generator.

        Args:
            model_key (str): The key for the desired model config. This config
                             should have the 'reasoning' flag enabled.
            include_test_input (bool): If True, includes the test input grid in the prompt.
            max_attempts (int): Maximum number of retry attempts for empty responses.
        """
        if model_key not in MODEL_CONFIGURATIONS:
            raise ValueError(f"Model key '{model_key}' not found.")
        
        self.model_key = model_key
        self.include_test_input = include_test_input
        self.include_train_input = include_train_input 
        self.include_abstraction = include_abstraction
        self.include_image = include_image
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.provider = LLMProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    def generate(
        self,
        task: ARCTask,
        abstractions: Optional[dict] = None,
    ) -> dict:
        """
        Generates an NL program by building a prompt and querying an LLM.
        """
        print(f"Generating instructions for task {task.task_id} with model {self.model_key}...")
        
        if self.include_image:
            user_content = []
            if self.include_train_input:
                for i, ex in enumerate(task.trainingExamples):
                    user_content.append({"type": "text", "text": f"\nExample {i+1}:\n"})
                    inp = grid_to_ascii(ex["input"])
                    out = grid_to_ascii(ex["output"])
                    
                    inp_text = f"Input Grid:\n{inp}\n"
                    if abstractions and self.include_abstraction:
                        inp_json = abstractions['train'][i]['input']
                        inp_text += f"Input Grid Json:\n{inp_json}\n"
                    
                    user_content.append({"type": "text", "text": inp_text})
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_from_grid(ex['input'])}"}})
                    
                    out_text = f"Output Grid:\n{out}\n"
                    if abstractions and self.include_abstraction:
                        out_json = abstractions['train'][i]['output']
                        out_text += f"Output Grid Json:\n{out_json}\n"
                        
                    user_content.append({"type": "text", "text": out_text})
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_from_grid(ex['output'])}"}})

            if self.include_test_input:
                test_inp = grid_to_ascii(task.testExamples[0]["input"])
                user_content.append({"type": "text", "text": f"\nTest Input Grid:\n\n{test_inp}"})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_from_grid(task.testExamples[0]['input'])}"}})
            
            messages = [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            # Trace for debug (simplified)
            system_prompt = GENERATOR_SYSTEM_PROMPT
            user_prompt = str(user_content)
            
        else:
            system_prompt, user_prompt = build_generator_prompt(
                task,
                abstractions=abstractions,
                include_train_input=self.include_train_input,
                include_test_input=self.include_test_input,
                include_abstraction=self.include_abstraction
            )
                
            messages = [
                {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]

        # Retry logic for empty responses
        max_attempts = self.max_attempts
        
        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"Warning: Empty instructions received. Retrying... (Attempt {attempt+1}/{max_attempts})")
            
            response = self.provider.chat(
                config=MODEL_CONFIGURATIONS[self.model_key],
                messages=messages
            )
            
            content, reasoning, usage_stats = self.provider.parse_response(response)
            
            # Check if content is valid
            if content and content.strip():
                return {
                    "instructions": content,
                    "reasoning": reasoning,
                    "usage": usage_stats,
                    "raw_response": content,
                    "model_used": self.model_key,
                    "prompt_trace": {
                        "system": system_prompt,
                        "user": user_prompt
                    },
                    "usage": usage_stats
                }
        
        # If all attempts fail
        print("Error: Failed to generate non-empty instructions after multiple attempts.")
        return {
            "instructions": "",
            "reasoning": reasoning if 'reasoning' in locals() else [],
            "usage": usage_stats if 'usage_stats' in locals() else None,
            "raw_response": content if 'content' in locals() else "",
            "model_used": self.model_key,
            "prompt_trace": {
                "system": system_prompt,
                "user": user_prompt
            },
            "status": "failed_empty_response"
        }
