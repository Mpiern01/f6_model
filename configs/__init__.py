"""
Config Module
Configuration loading and management
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Handle extends
    if "extends" in config:
        base_path = Path(config_path).parent / config["extends"]
        base_config = load_config(str(base_path))
        # Merge: base config first, then override with current
        merged = {**base_config, **config}
        # Remove extends key
        merged.pop("extends", None)
        return merged
    
    return config

