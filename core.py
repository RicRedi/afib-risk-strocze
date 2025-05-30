# -*- coding: utf-8 -*-

"""
Created on 30. 05. 2025 at 15:18:43

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
    
"""
import inspect
from typing import Callable, Dict

def validate_inputs_from_signature(
    func: Callable = None,
    locals_dict: Dict = None,
    ) -> None:
    """
    Validates that all required arguments from a function's signature are present
    in the provided locals dictionary.
    
    Args:
        func (callable): The function whose signature will be validated.
        locals_dict (dict): A dictionary representing the local variables
                            to check for required arguments.

    Raises:
        ValueError: If any required argument (without a default value) is missing from locals_dict.
    """
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and name not in locals_dict:
            raise ValueError(f"Missing required argument: {name}")
