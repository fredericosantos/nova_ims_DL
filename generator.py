# %%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
import os
import time
from utils import ReflectionPadding2D, initializer
from blocks import *

def Generator(
    filters: int = 64,
    input_shape: int = 256,
    input_filters: int = 3,
    n_convBlocks: int = 2,
    n_resBlocks: int = 6,
    n_deconvBlocks: int = 2,
    name=None,
):
    """Generator for CycleGAN
    Args:
        filters: number of filters to use on the first convolution"""

    img_input = tf.keras.layers.Input(
        shape=[input_shape, input_shape, input_filters], name="Input")

    # Because the kernel size is 7x7 on the first convolution, we add reflective padding 3x3
    x = ReflectionPadding2D(padding=(3, 3), name="REF_0")(img_input)

    # First 7x7, convolution c7s1_64
    x = convBlock(filters=filters, activation="relu", strides=1, 
                  kernel_size=7, norm_type="instance", padding="valid", name=f"c7s1-{filters}")(x)

    # Convolution / Encoder
    for _ in range(n_convBlocks):
        filters *= 2
        x = convBlock(filters=filters, activation="relu",
                      norm_type="instance", name=f"d{filters}")(x)

    # Residual Blocks
    for _ in range(n_resBlocks):
        x = resBlock(x, activation="relu", norm_type="instance",
                     name=f"R{x.shape[-1]}_block{_}")

    for _ in range(n_deconvBlocks):
        filters //= 2
        x = deconvBlock(filters=filters, activation="relu",
                        norm_type="instance", name=f"u{filters}")(x)

    x = ReflectionPadding2D((3, 3), name="REF_last")(x)
    x = convBlock(filters=input_filters, activation="tanh", strides=1,
                  kernel_size=7, padding="valid", name=f"c7s1-{input_filters}")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


# generator = Generator()
# tf.keras.utils.plot_model(generator, to_file="generator.png", show_shapes=True, dpi=128)

# %%
