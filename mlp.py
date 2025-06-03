# -*- coding: utf-8 -*-

"""
Created on 02. 06. 2025 at 18:38:57

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
    Module containing a Multi-Layer Perceptron (MLP) class for regression tasks.
    This class implements a simple MLP with one hidden layer and uses the Adam optimizer.
    It includes methods for training, predicting, and evaluating the model.
"""
import torch as t
from torch import nn
from utils.config_singleton import ConfigSingleton

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}

class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) for regression tasks.
    This class implements a simple MLP with one hidden layer and uses the Adam optimizer.
    It includes methods for training, predicting, and evaluating the model.
    Attributes:
        cfg (Config): Configuration object containing model parameters.
        model (nn.Sequential): Sequential model containing the MLP layers.
    """
    def __init__(
        self,
        ):
        super().__init__()
        self.cfg = ConfigSingleton.get()
        layers = []
        in_size = len(self.cfg.variables.model_features)
        if self.cfg.model.architecture.hidden_sizes is not None:
            for h in self.cfg.model.architecture.hidden_sizes:
                layers.append(
                    nn.Linear(
                        in_size,
                        h
                        ))
                layers.append(
                    ACTIVATIONS[self.cfg.model.architecture.activation]
                    )
                layers.append(
                    nn.Dropout(self.cfg.model.architecture.dropout)
                    )
                in_size = h
        layers.append(
            nn.Linear(in_size, self.cfg.model.architecture.output_size)
            )
        layers.append(
                ACTIVATIONS['sigmoid']  # Output layer activation for binary classification
                )
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        x: t.Tensor,
        ) -> t.Tensor:
        """Forward pass through the MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.model(x)
