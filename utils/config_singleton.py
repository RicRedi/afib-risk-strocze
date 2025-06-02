# -*- coding: utf-8 -*-

"""
Created on 02. 06. 2025 at 17:07:31

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
    Singleton class to hold the configuration object.
    This ensures that the configuration is loaded only once
    and can be accessed globally.
"""
from utils.load_config import load_config_as_object

class ConfigSingleton:
    """
    Singleton class to hold the configuration object.
    This ensures that the configuration is loaded only once
    and can be accessed globally.
    """
    _instance = None

    @classmethod
    def set(
        cls,
        ) -> None:
        """
        Sets the singleton instance of the configuration object.
        This method loads the configuration from a YAML file
        and initializes the singleton instance.
        """
        cls._instance = load_config_as_object('config/config.yaml')

    @classmethod
    def get(
        cls
        ) -> 'ConfigSingleton':
        """
        Returns the singleton instance of the configuration object.
        """
        if cls._instance is None:
            cls._instance = ConfigSingleton()
        return cls._instance
