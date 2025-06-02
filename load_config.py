# -*- coding: utf-8 -*-

"""
Created on 02. 06. 2025 at 15:36:22

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Short description of the script.
"""
import os
import yaml

class ConfigObject:
    """
    A class to represent a configuration object loaded from a YAML file.
    This class allows for dynamic attribute access based on the keys
    in the configuration dictionary.
    """
    def __init__(
        self,
        config_dict,
        base_path=None,
        ) -> None:
        """
        Initialize the ConfigObject with a dictionary.
        Args:
            config_dict (dict): A dictionary containing configuration parameters.
        """
        if not isinstance(config_dict, dict):
            raise TypeError("config_dict must be a dictionary")
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = ConfigObject(value)
            elif (
                isinstance(value, str) and
                value.endswith(('.yaml', '.yml')) and
                os.path.exists(os.path.join(base_path or '', value))
                ):
                # Load nested config file
                nested_path = os.path.join(base_path or '', value)
                with open(nested_path, 'r', encoding='utf-8') as f:
                    nested_data = yaml.safe_load(f)
                value = ConfigObject(nested_data, base_path=os.path.dirname(nested_path))
            setattr(self, key, value)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

def load_config_as_object(
    path: str
    ) -> ConfigObject:
    """
    Load a YAML configuration file and return it as a ConfigObject.
    Args:
        path (str): Path to the YAML configuration file.
    Returns:
        ConfigObject: An object containing the configuration parameters.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(
        path,
        'r',
        encoding='utf-8',
        ) as file:
        config_dict = yaml.safe_load(file)
    return ConfigObject(config_dict)
