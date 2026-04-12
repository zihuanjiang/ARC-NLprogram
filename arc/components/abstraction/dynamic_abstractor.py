# abstraction/dynamic_abstractor.py
import os
import json
from typing import Optional, Dict, Any, List
from .llm_abstractor import LLMAbstractor, MODEL_CONFIGURATIONS
from arc.data.ARCTask import ARCTask
from .dynamic_prompts import (
    SCHEMA_GENERATION_PROMPT,
    SCHEMA_GENERATION_TEMPLATE,
    TRAIN_PAIR_ABSTRACT_PROMPT_DYNAMIC,
    TEST_ABSTRACT_PROMPT_DYNAMIC,
    GeneratedGridObjectSchema
)
from .v1_prompts import Grid, GridPair, Coordinate, BoundingBox

from arc.solvers.registry import AbstractorRegistry

@AbstractorRegistry.register("dynamic")
class DynamicAbstractor(LLMAbstractor):
    """
    Implementation of Abstractor that dynamically generates a schema based on training examples,
    then uses that schema to abstract grids.
    """
    
    def __init__(
        self,
        model_key: str,
        include_train_input: bool = True,
        include_test_input: bool = True,
        schema_model_key: Optional[str] = None,
        **kwargs, #TODO: Add include_image
    ):
        """
        Args:
            model_key: Model key for grid abstraction
            include_train_input: Generate grid abstraction for training pairs
            include_test_input: Generate grid abstraction for test grid
            schema_model_key: Optional different model key for schema generation.
                            If None, uses the same model_key.
        """
        super().__init__(model_key=model_key, include_train_input=include_train_input, include_test_input=include_test_input)
        self.schema_model_key = schema_model_key or model_key
        self._cached_schema: Optional[Dict[str, Any]] = None
        self._cached_schema_for_task: Optional[str] = None
    
    def _generate_schema_from_training(self, task: ARCTask) -> Dict[str, Any]:
        """
        Generate a custom schema based on training examples using structured output API.
        
        Returns:
            A JSON schema dict that can be used for structured output
        """
        # Build content with all training examples
        content = []
        
        for i, train_pair in enumerate(task.trainingExamples):
            input_grid = train_pair['input']
            output_grid = train_pair['output']
            input_grid_base64 = self.base64_from_grid(input_grid)
            output_grid_base64 = self.base64_from_grid(output_grid)
            
            content.append({
                "type": "text",
                "text": f"Training Example {i+1}:\n"
                       f"Input Grid (matrix): {input_grid}\n"
                       f"Input Grid (png): "
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{input_grid_base64}"}
            })
            content.append({
                "type": "text",
                "text": f"Output Grid (matrix): {output_grid}\n"
                       f"Output Grid (png): "
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{output_grid_base64}"}
            })
        
        messages = [
            {"role": "system", "content": SCHEMA_GENERATION_PROMPT},
            {"role": "user", "content": content}
        ]
        
        # Use structured output to generate schema
        schema_template = SCHEMA_GENERATION_TEMPLATE
        model_config = MODEL_CONFIGURATIONS[self.schema_model_key]
        if model_config.extra_body is None:
            model_config.extra_body = {}
        
        model_config.extra_body.update({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    **schema_template,
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
            
        generated_schema_data = json.loads(raw_response)
        
        # Convert GeneratedSchema to actual JSON schema format
        json_schema = self._convert_to_json_schema(generated_schema_data)
        return json_schema
    
    def _convert_to_json_schema(self, generated_schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the GeneratedGridObjectSchema format to a usable JSON schema.
        Merges the dynamically generated GridObject with fixed Grid and GridPair schemas.
        
        Args:
            generated_schema_data: The GeneratedGridObjectSchema dict from LLM
            
        Returns:
            A JSON schema dict with definitions for Coordinate, BoundingBox, GridObject, Grid, and GridPair
        """
        def build_property_schema(field_def: Dict[str, Any]) -> Dict[str, Any]:
            """Convert a field definition to JSON schema property."""
            field_type = field_def.get("type", "string")
            prop_schema = {"type": field_type}
            
            if "description" in field_def:
                prop_schema["description"] = field_def["description"]
            
            if field_type == "array":
                items_type = field_def.get("items_type", "string")
                if items_type == "object":
                    # Handle array of objects - check if it's Coordinate
                    # For now, assume it's a reference or inline object
                    if "properties" in field_def:
                        prop_schema["items"] = {
                            "type": "object",
                            "properties": {
                                k: build_property_schema(v) for k, v in field_def["properties"].items()
                            }
                        }
                    else:
                        # Default to Coordinate structure
                        prop_schema["items"] = {"$ref": "#/definitions/Coordinate"}
                else:
                    prop_schema["items"] = {"type": items_type}
            elif field_type == "object":
                if "properties" in field_def and field_def["properties"]:
                    prop_schema["properties"] = {
                        k: build_property_schema(v) for k, v in field_def["properties"].items()
                    }
                    prop_schema["required"] = [
                        k for k, v in field_def["properties"].items()
                        if v.get("required", False)
                    ]
                else:
                    # Default to BoundingBox structure
                    prop_schema = {"$ref": "#/definitions/BoundingBox"}
            
            return prop_schema
        
        # Get the fixed schemas from v1_prompts
        coordinate_schema = Coordinate.model_json_schema()
        bounding_box_schema = BoundingBox.model_json_schema()
        grid_schema = Grid.model_json_schema()
        grid_pair_schema = GridPair.model_json_schema()
        
        # Extract GridObject fields from generated schema
        grid_object_fields = generated_schema_data.get("grid_object_fields", [])
        
        # Build GridObject schema from generated fields
        grid_object_properties = {}
        grid_object_required = []
        
        for field_def in grid_object_fields:
            field_name = field_def.get("name")
            if not field_name:
                continue
            
            # Special handling for known fields
            if field_name == "pixels":
                # Pixels should be array of Coordinate
                grid_object_properties["pixels"] = {
                    "type": "array",
                    "description": field_def.get("description", "Exhaustive list of all object pixels."),
                    "items": {"$ref": "#/definitions/Coordinate"},
                    "minItems": 1
                }
                if field_def.get("required", True):
                    grid_object_required.append("pixels")
            elif field_name == "bounding_box":
                # BoundingBox should reference the definition
                grid_object_properties["bounding_box"] = {
                    "$ref": "#/definitions/BoundingBox",
                    "description": field_def.get("description", "Smallest box enclosing the object.")
                }
                if field_def.get("required", True):
                    grid_object_required.append("bounding_box")
            else:
                # Build property schema for other fields
                grid_object_properties[field_name] = build_property_schema(field_def)
                if field_def.get("required", False):
                    grid_object_required.append(field_name)
        
        # Build complete GridObject schema
        grid_object_schema = {
            "type": "object",
            "description": generated_schema_data.get("description", "A distinct contiguous or patterned set of pixels."),
            "properties": grid_object_properties,
            "required": grid_object_required,
            "additionalProperties": False
        }
        
        # Update Grid schema to reference our GridObject
        # The Grid schema from Pydantic already has the right structure, we just need to update the reference
        if "properties" in grid_schema and "objects" in grid_schema["properties"]:
            if "items" in grid_schema["properties"]["objects"]:
                grid_schema["properties"]["objects"]["items"] = {"$ref": "#/definitions/GridObject"}
        
        # Update GridPair schema to reference our Grid
        if "properties" in grid_pair_schema:
            grid_pair_schema["properties"]["input"] = {"$ref": "#/definitions/Grid"}
            grid_pair_schema["properties"]["output"] = {"$ref": "#/definitions/Grid"}
        
        # Build the complete JSON schema with all definitions
        json_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "definitions": {
                "Coordinate": coordinate_schema,
                "BoundingBox": bounding_box_schema,
                "GridObject": grid_object_schema,
                "Grid": grid_schema,
                "GridPair": grid_pair_schema
            }
        }
        
        return json_schema
    
    def _get_or_generate_schema(self, task: ARCTask) -> Dict[str, Any]:
        """Get cached schema or generate a new one for this task."""
        task_id = task.task_id if hasattr(task, 'task_id') else str(id(task))
        
        if self._cached_schema is None or self._cached_schema_for_task != task_id:
            print(f"Generating custom schema for task {task_id}...")
            self._cached_schema = self._generate_schema_from_training(task)
            self._cached_schema_for_task = task_id
            print(f"Schema generation complete.")
        
        return self._cached_schema
    
    def _inline_schema_refs(self, schema: Dict[str, Any], definitions: Dict[str, Any]) -> Dict[str, Any]:
        """Inline $ref references in a schema."""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"]
                if ref_path.startswith("#/definitions/"):
                    ref_name = ref_path.split("/")[-1]
                    if ref_name in definitions:
                        return self._inline_schema_refs(definitions[ref_name], definitions)
            else:
                return {
                    k: self._inline_schema_refs(v, definitions) if isinstance(v, (dict, list)) else v
                    for k, v in schema.items()
                }
        elif isinstance(schema, list):
            return [self._inline_schema_refs(item, definitions) if isinstance(item, (dict, list)) else item for item in schema]
        return schema
    
    def abstract_train_pairs(
        self,
        task: ARCTask,
    ) -> tuple[list[dict], list]:
        """Abstract training pairs using the dynamically generated schema."""
        # Get or generate schema
        json_schema = self._get_or_generate_schema(task)
        definitions = json_schema.get("definitions", {})
        
        # Inline references in GridPair schema
        grid_pair_schema = self._inline_schema_refs(definitions.get("GridPair", {}), definitions)
        grid_pair_schema["title"] = "GridPair"
        
        inputs = [train_pair['input'] for train_pair in task.trainingExamples]
        outputs = [train_pair['output'] for train_pair in task.trainingExamples]
        train_abstraction = []
        all_usage = []
        
        for i, (input_grid, output_grid) in enumerate(zip(inputs, outputs)):
            input_grid_base64 = self.base64_from_grid(input_grid)
            output_grid_base64 = self.base64_from_grid(output_grid)
            
            # Generate description (similar to v1)
            messages = [
                {"role": "system", "content": TRAIN_PAIR_ABSTRACT_PROMPT_DYNAMIC},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Custom Schema:\n{json.dumps(json_schema, indent=2)}\n\n"},
                        {"type": "text", "text": f"Input Grid (matrix): {input_grid}\nInput Grid (png): "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{input_grid_base64}"}},
                        {"type": "text", "text": f"Output Grid (matrix): {output_grid}\nOutput Grid (png): "},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{output_grid_base64}"}}
                    ]
                }
            ]
            
            model_config = MODEL_CONFIGURATIONS[self.model_key]
            if model_config.extra_body is None:
                model_config.extra_body = {}
            
            model_config.extra_body.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        **grid_pair_schema,
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

            parsed = json.loads(raw_response)
            train_abstraction.append(parsed)
            
            _, _, usage = self.provider.parse_response(response)
            all_usage.append(usage)
        return train_abstraction, all_usage
    
    def abstract_test_grids(
        self,
        task: ARCTask,
        grid_abstraction: Optional[Dict] = None,
    ) -> tuple[list[dict], list]:
        """Abstract test grids using the dynamically generated schema."""
        # Get or generate schema
        json_schema = self._get_or_generate_schema(task)
        definitions = json_schema.get("definitions", {})
        
        # Inline references in Grid schema
        grid_schema = self._inline_schema_refs(definitions.get("Grid", {}), definitions)
        grid_schema["title"] = "Grid"
        
        train_inputs = [train_pair['input'] for train_pair in task.trainingExamples]
        train_inputs_base64 = [self.base64_from_grid(train_input_grid) for train_input_grid in train_inputs]
        
        sample_content = [
            {"type": "text", "text": f"Custom Schema:\n{json.dumps(json_schema, indent=2)}\n\n"}
        ]
        
        if grid_abstraction:
            for idx, (train_input, train_input_base64, train_grid_abstraction) in enumerate(
                zip(train_inputs, train_inputs_base64, grid_abstraction.get('train', []))
            ):
                sample_content.append({
                    "type": "text",
                    "text": f"Sample Input Grid {idx} (matrix): {train_input}\nSample Input Grid {idx} (png): "
                })
                sample_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{train_input_base64}"}
                })
                sample_content.append({
                    "type": "text",
                    "text": f"Sample Output Abstraction {idx}: {json.dumps(train_grid_abstraction.get('input', {}), indent=2)}"
                })
        
        test_abstraction = []
        all_usage = []
        for i, test_example in enumerate(task.testExamples):
            input_grid = test_example['input']
            input_grid_base64 = self.base64_from_grid(input_grid)
            
            test_content = sample_content + [
                {"type": "text", "text": f"Your Input Grid (matrix): {input_grid}\nYour Input Grid (png): "},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{input_grid_base64}"}}
            ]
            
            messages = [
                {"role": "system", "content": TEST_ABSTRACT_PROMPT_DYNAMIC},
                {"role": "user", "content": test_content}
            ]
            
            model_config = MODEL_CONFIGURATIONS[self.model_key]
            if model_config.extra_body is None:
                model_config.extra_body = {}
            
            model_config.extra_body.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        **grid_schema,
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

            parsed = json.loads(raw_response)
            test_abstraction.append(parsed)
            
            _, _, usage = self.provider.parse_response(response)
            all_usage.append(usage)

        grid_abstraction.update({"test": test_abstraction})
        return grid_abstraction, all_usage
