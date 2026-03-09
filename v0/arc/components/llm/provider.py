from openai import OpenAI
from .config import ModelConfig

class LLMProvider:
    def __init__(self, api_key, base_url="https://openrouter.ai/api/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, config: ModelConfig, messages: list):
        extra_body = config.extra_body.copy()

        response = self.client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            extra_body=extra_body  
        )
        return response

    @staticmethod
    def parse_response(response):
        """
        Parses the LLM response to extract content, reasoning, and usage stats.
        """
        content = response.choices[0].message.content.strip()
        reasoning = None

        # Dump to dict to see all fields
        response_dict = response.model_dump()
        message_dict = response_dict.get('choices', [{}])[0].get('message', {})

        # Check all known locations for reasoning
        if 'reasoning_details' in message_dict and message_dict['reasoning_details']:
            reasoning = message_dict['reasoning_details']
        elif 'reasoning' in message_dict and message_dict['reasoning']:
            reasoning = message_dict['reasoning']
        elif 'reasoning_content' in message_dict and message_dict['reasoning_content']:
            # 'reasoning_content' is the standard for OpenAI's o1 models
            reasoning = message_dict['reasoning_content']

        # Fallback: Check if it leaked into the content with <think> tags
        if not reasoning and "<think>" in content:
            parts = content.split("</think>")
            if len(parts) > 1:
                reasoning = parts[0].replace("<think>", "").strip()
                content = parts[1].strip()

        # Extract usage
        usage = response.usage
        usage_stats = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "model": response.model
        }

        return content, reasoning, usage_stats