# abstraction/llm_abstractor.py
import os
import json
from typing import Optional
from .llm_abstractor import LLMAbstractor
from arc.data.ARCTask import ARCTask
from arc.components.llm.config import MODEL_CONFIGURATIONS
from .v1_prompts import Grid, GridPair, TRAIN_PAIR_DESCRIBE_PROMPT, TRAIN_PAIR_ABSTRACT_PROMPT, TEST_ABSTRACT_PROMPT

from arc.solvers.registry import AbstractorRegistry

@AbstractorRegistry.register("v1")
class LLMAbstractor_v1(LLMAbstractor):
    """
    Implementation of Abstractor that uses LLM to generate grid abstractions.
    """
    def __init__(self, model_key: str, include_train_input: bool = True, include_test_input: bool = True, include_image: bool = True, max_tokens: Optional[int] = None, temperature: Optional[float] = None):
        super().__init__(model_key=model_key, include_train_input=include_train_input, include_test_input=include_test_input)
        self.include_image = include_image
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def _generate_grid_description(
        self, input_grid, output_grid, input_grid_base64, output_grid_base64
    ):
        content = [
            {"type": "text", "text": f"Input Grid (matrix): {input_grid}\nInput Grid (png): "},
        ]
        if self.include_image:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{input_grid_base64}"}})
            
        content.append({"type": "text", "text": f"Output Grid (matrix): {output_grid}\nOutput Grid (png): "})
        
        if self.include_image:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{output_grid_base64}"}})

        messages = [
            {"role": "system", "content": TRAIN_PAIR_DESCRIBE_PROMPT},
            {
                "role": "user", 
                "content": content
            }
        ]
        model_config = MODEL_CONFIGURATIONS[self.model_key]
        if self.max_tokens: model_config.max_tokens = self.max_tokens
        if self.temperature: model_config.temperature = self.temperature
        # Clear response_format for natural language description (not JSON)
        if model_config.extra_body and 'response_format' in model_config.extra_body:
            model_config.extra_body = {k: v for k, v in model_config.extra_body.items() if k != 'response_format'}
        
        response = self.provider.chat(config=model_config, messages=messages)
        raw_response = response.choices[0].message.content
        
        _, _, usage = self.provider.parse_response(response)
        return raw_response, usage

    def abstract_train_pairs(
        self,
        task: ARCTask,
    ) -> tuple[list[dict], list[dict]]:
        inputs = [train_pair['input'] for train_pair in task.trainingExamples]
        outputs = [train_pair['output'] for train_pair in task.trainingExamples]
        train_abstraction = []
        all_usage = []
        
        for i, (input_grid, output_grid) in enumerate(zip(inputs, outputs)):
            input_grid_base64 = self.base64_from_grid(input_grid)
            output_grid_base64 = self.base64_from_grid(output_grid)
            paired_grid_description, usage1 = self._generate_grid_description(input_grid, output_grid, input_grid_base64, output_grid_base64)
            all_usage.append(usage1)

            content = [
                {"type": "text", "text": f"Input Grid (matrix): {input_grid}\nInput Grid (png): "},
            ]
            if self.include_image:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{input_grid_base64}"}})
            
            content.append({"type": "text", "text": f"Output Grid (matrix): {output_grid}\nOutput Grid (png): "})
            
            if self.include_image:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{output_grid_base64}"}})
            
            content.append({"type": "text", "text": f"Grid descriptions: {paired_grid_description}"})

            messages = [
                {"role": "system", "content": TRAIN_PAIR_ABSTRACT_PROMPT},
                {
                    "role": "user", 
                    "content": content
                }
            ]
            schema = GridPair.model_json_schema()
            model_config = MODEL_CONFIGURATIONS[self.model_key]
            if self.max_tokens: model_config.max_tokens = self.max_tokens
            if self.temperature: model_config.temperature = self.temperature
            if model_config.extra_body is None:
                model_config.extra_body = {}

            model_config.extra_body.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                    **schema,
                    "strict": True,
                    "additionalProperties": False
                    }
                }
            })
            response = self.provider.chat(config=model_config, messages=messages)
            raw_response = response.choices[0].message.content
            
            # Clean markdown code blocks if present
            if raw_response and "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif raw_response and "```" in raw_response:
                raw_response = raw_response.split("```")[1].split("```")[0].strip()
            
            # Handle empty responses from LLM
            if not raw_response or raw_response.strip() == "":
                print(f"Warning: Empty response from LLM for train pair {i}")
                print(f"Usage info: {response.usage if hasattr(response, 'usage') else 'No usage info'}")
                # Return empty dict to continue processing
                parsed = {}
            else:
                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for train pair {i}")
                    print(f"Raw response: {raw_response[:200]}...")
                    print(f"Error: {e}")
                    parsed = {}
            
            train_abstraction.append(parsed)
            
            _, _, usage2 = self.provider.parse_response(response)
            all_usage.append(usage2)
            
        return train_abstraction, all_usage
            
    def abstract_test_grids(
        self,
        task: ARCTask,
        grid_abstraction: Optional[dict] = None,
    ) -> tuple[list[dict], list[dict]]:
        train_inputs = [train_pair['input'] for train_pair in task.trainingExamples]
        train_inputs_base64 = [self.base64_from_grid(train_input_grid) for train_input_grid in train_inputs]
        sample_content = []
        if grid_abstraction:
            for idx, (train_input, train_input_base64, train_grid_abstraction) in enumerate(zip(train_inputs, train_inputs_base64, grid_abstraction['train'])):
                sample_content.append({"type": "text", "text": f"Sample Input Grid {idx} (matrix): {train_input}\nSample Input Grid {idx} (png): "})
                if self.include_image:
                    sample_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{train_input_base64}"}})
                if 'input' in train_grid_abstraction:
                    sample_content.append({"type": "text", "text": f"Sample Output Abstraction {idx}: {train_grid_abstraction['input']}"})
                else:
                    sample_content.append({"type": "text", "text": f"Sample Output Abstraction {idx}: (Abstraction failed generation)"})
        else:
            grid_abstraction = {}
        
        test_abstraction = []
        all_usage = []
        
        for i, input_grid in enumerate(task.testExamples):
            input_grid = input_grid['input']
            input_grid_base64 = self.base64_from_grid(input_grid)

            sample_content.append({"type": "text", "text": f"Your Input Grid (matrix): {input_grid}\nYour Input Grid (png): "})
            if self.include_image:
                sample_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{input_grid_base64}"}})

            messages = [
                {"role": "system", "content": TEST_ABSTRACT_PROMPT},
                {
                    "role": "user", 
                    "content": sample_content
                }
            ]
            schema = Grid.model_json_schema()
            model_config = MODEL_CONFIGURATIONS[self.model_key]
            if self.max_tokens: model_config.max_tokens = self.max_tokens
            if self.temperature: model_config.temperature = self.temperature
            if model_config.extra_body is None:
                model_config.extra_body = {}

            model_config.extra_body.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                    **schema,
                    "strict": True,
                    "additionalProperties": False
                    }
                }
            })
            response = self.provider.chat(config=model_config, messages=messages)
            raw_response = response.choices[0].message.content
            
            # Clean markdown code blocks if present
            if raw_response and "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif raw_response and "```" in raw_response:
                raw_response = raw_response.split("```")[1].split("```")[0].strip()
            
            # Handle empty responses from LLM
            if not raw_response or raw_response.strip() == "":
                print(f"Warning: Empty response from LLM for test grid {i}")
                print(f"Usage info: {response.usage if hasattr(response, 'usage') else 'No usage info'}")
                parsed = {}
            else:
                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for test grid {i}")
                    print(f"Raw response: {raw_response[:200]}...")
                    print(f"Error: {e}")
                    parsed = {}
            
            test_abstraction.append(parsed)
            
            _, _, usage = self.provider.parse_response(response)
            all_usage.append(usage)
            
        grid_abstraction.update({"test": test_abstraction})
        return grid_abstraction, all_usage