# %%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
import os
import time
import utils


def initializer(mean=0.0, stddev=0.02, seed=None):
    return tf.random_normal_initializer(mean, stddev, seed)


def convBlock(
    filters,
    activation: "ReLU / LeakyReLU",
    kernel_size: int = 3,
    norm_type: "instance/batch/none" = None,
    dropout: "0 to 1" = 0
):
    """
    Args:
        activation: ReLU or LeakyReLU
        filters: Integer, the dimensionality of the output space (i.e. the number of
        output filters in the convolution).
        kernel_size: Single integer to specify the same value for all spatial dimensions.
        norm_type: choose what type of normalization to apply as per the paper.
    """

    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(2, 2),
        padding="same",
        kernel_initializer=initializer(),
        use_bias=False,
    ))
    if norm_type.lower() == "batch":
        block.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == "instance":
        block.add(InstanceNormalization())
    if dropout > 0:
        block.add(tf.keras.layers.Dropout)
    if activation.lower() == "relu":
        block.add(keras.layers.Activation("relu"))
    elif activation.lower() == "leakyrelu":
        block.add(keras.layers.Activation(tf.nn.leaky_relu))
    return block


def deconvBlock(
    filters,
    activation: "ReLU / LeakyReLU",
    kernel_size: int = 3,
    norm_type: "instance/batch/none" = None,
    dropout: "0 to 1" = 0
):
    """
    Args:
        activation: ReLU or LeakyReLU
        filters: Integer, the dimensionality of the output space (i.e. the number of
        output filters in the deconvolution).
        kernel_size: Single integer to specify the same value for all spatial dimensions.
        norm_type: choose what type of normalization to apply as per the paper.
    """

    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(kernel_size, kernel_size),
        strides=(2, 2),
        padding="same",
        kernel_initializer=initializer(),
        use_bias=False,
    ))
    if norm_type.lower() == "batch":
        block.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == "instance":
        block.add(InstanceNormalization())
    if dropout > 0:
        block.add(tf.keras.layers.Dropout)
    if activation.lower() == "relu":
        block.add(keras.layers.Activation("relu"))
    elif activation.lower() == "leakyrelu":
        block.add(keras.layers.Activation(tf.nn.leaky_relu))
    return block


# %%
def Generator(
    input_shape: int, 
    input_filters: int,
    name=None,
):
    """Generator for CycleGAN
    Args:
        input_shape: integer of image weight or height (must be square)
        input_filters: number of filters to use on the first convolution"""
    inputs = tf.keras.layers.Input(
        shape=[input_shape, input_shape, input_filters])
    
    # Because the kernel size is 7x7 on the first convolution, we add reflective padding 3x3
    reflectivePadding = utils.ReflectionPadding2D(padding=(3,3))
    
    # First 7x7, convolution c7s1_64
    c7s1_64 = convBlock(input_filters, "relu", kernel_size=7, norm_type="instance")

    # Two convolution blocks