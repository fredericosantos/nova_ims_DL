import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
import os
import time
from utils import ReflectionPadding2D, initializer


# Encoder / Convolutional Block
def convBlock(
    filters,
    activation: "ReLU / LeakyReLU / tanh",
    norm_type: "instance/batch/none" = None,
    dropout: "0 to 1" = 0,
    kernel_size: int = 3,
    padding: "valid/same" = "same",
    strides: int = 2,
    name=None,
):
    """
    Args:
        activation: ReLU or LeakyReLU or tanh
        filters: Integer, the dimensionality of the output space (i.e. the number of
        output filters in the convolution).
        kernel_size: Single integer to specify the same value for all spatial dimensions.
        norm_type: choose what type of normalization to apply as per the paper.
        padding: valid applies no padding, same leaves the tensor the same size after the kernel pass
    """

    block = keras.Sequential(name=name)
    block.add(keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=initializer(),
        use_bias=False,
        name=name, 
    ))
    if norm_type:
        if norm_type.lower() == "batch":
            block.add(keras.layers.BatchNormalization())
        elif norm_type.lower() == "instance":
            block.add(InstanceNormalization())
    if dropout > 0:
        block.add(keras.layers.Dropout(dropout))
    if activation.lower() == "relu":
        block.add(keras.layers.ReLU())
    elif activation.lower() == "leakyrelu":
        block.add(keras.layers.LeakyReLU(0.2))
    elif activation.lower() == "tanh":
        block.add(keras.layers.Activation("tanh"))
    return block


# Decoder / Deconvolutional Block
def deconvBlock(
    filters,
    activation: "ReLU / LeakyReLU",
    norm_type: "instance/batch/none" = None,
    dropout: "0 to 1" = 0,
    kernel_size: int = 3,
    padding: "valid/same" = "same",
    strides: int = 2,
    name=None,
):
    """
    Args:
        activation: ReLU or LeakyReLU
        filters: Integer, the dimensionality of the output space (i.e. the number of
        output filters in the deconvolution).
        kernel_size: Single integer to specify the same value for all spatial dimensions.
        norm_type: choose what type of normalization to apply as per the paper.
    """

    block = keras.Sequential(name=name)
    block.add(keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=initializer(),
        use_bias=False,
        name=name,
    ))
    if norm_type:
        if norm_type.lower() == "batch":
            block.add(keras.layers.BatchNormalization())
        elif norm_type.lower() == "instance":
            block.add(InstanceNormalization())
    if dropout > 0:
        block.add(keras.layers.Dropout)
    if activation.lower() == "relu":
        block.add(keras.layers.ReLU())
    elif activation.lower() == "leakyrelu":
        block.add(keras.layers.LeakyReLU(0.2))
    return block

# Residual Network Block
def resBlock(
    x: "Tensor",
    activation: "ReLU / LeakyReLU",
    norm_type: "instance/batch/none" = None,
    dropout: "0 to 1" = 0,
    kernel_size: int = 3,
    name=None,
):
    filters = x.shape[-1]
    f = keras.Sequential(name=name)
    for i in range(2):
        init = tf.random_normal_initializer(0.0, 0.02)
        f.add(ReflectionPadding2D(name=f"REF_{i+1}")) # do not add padding to the Conv2D since we're adding padding here
        f.add(keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=1,
            kernel_initializer=init,
            padding="valid",
            use_bias=False,
            name=f"{name}_{i}", 
        ))
        if norm_type:
            if norm_type.lower() == "batch":
                f.add(keras.layers.BatchNormalization())
            elif norm_type.lower() == "instance":
                f.add(InstanceNormalization())
        if dropout > 0:
            f.add(keras.layers.Dropout)
        
        # According to the paper, 
        if i == 0:   
            if activation.lower() == "relu":
                f.add(keras.layers.ReLU())
            elif activation.lower() == "leakyrelu":
                f.add(keras.layers.LeakyReLU(0.2))
    
    x = keras.layers.add([f(x), x], name=f"Add_{name}")
    return x