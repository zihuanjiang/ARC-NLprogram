import os
import yaml
from copy import deepcopy

def deep_merge(base_dict, update_dict):
    """
    Deep merge update_dict into base_dict.
    Nested dictionaries are merged recursively.
    """
    result = deepcopy(base_dict)
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def load_config_with_updates(config_path, config_updates=None):
    """
    Load base config from YAML file and apply updates.
    
    Args:
        config_path: Full path to the base config YAML file
        config_updates: Dictionary of config updates to apply (optional)
    
    Returns:
        Merged configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    if config_updates:
        base_config = deep_merge(base_config, config_updates)
    
    return base_config
