"""
PyTorch & Keras Neural Network Architecture Module.

Provides modular PyTorch MLP architecture and equivalent Keras model builder.
"""

from typing import List
import torch
import torch.nn as nn


class PyTorchMLP(nn.Module):
    """Modular Multi-Layer Perceptron architecture in PyTorch."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_keras_mlp_equivalent(
    input_dim: int,
    hidden_dims: List[int] = [64, 32],
    output_dim: int = 1,
    dropout_rate: float = 0.2,
) -> str:
    """Return string representation of Keras functional equivalent architecture code."""
    keras_code = f"""
# Keras Functional API Equivalent Model
import tensorflow as tf
from tensorflow import keras
from keras import layers

inputs = keras.Input(shape=({input_dim},))
x = inputs
"""
    for h_dim in hidden_dims:
        keras_code += f"x = layers.Dense({h_dim}, activation='relu')(x)\n"
        keras_code += f"x = layers.BatchNormalization()(x)\n"
        if dropout_rate > 0:
            keras_code += f"x = layers.Dropout({dropout_rate})(x)\n"

    keras_code += f"outputs = layers.Dense({output_dim}, activation='sigmoid')(x)\n"
    keras_code += f"model = keras.Model(inputs=inputs, outputs=outputs)\n"
    return keras_code
