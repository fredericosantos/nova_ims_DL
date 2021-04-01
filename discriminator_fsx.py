# %%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
import os
import time
from utils import ReflectionPadding2D, initializer
from blocks import convBlock


def Discriminator(
    filters: int = 64,
    input_shape: int = 256,
    input_filters: int = 3,
    n_convBlocks: int = 3,
    name=None,
):
    """Discriminator for CycleGAN
    Args:
        filters: number of filters to use on the first convolution"""

    img_input = tf.keras.layers.Input(
        shape=[input_shape, input_shape, input_filters], name="Input"
    )
    # x = keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 127.5, offset=-1)
    # According to CycleGAN paper architecture, do not apply InstanceNormalization to first layer C64
    x = convBlock(
        filters=filters,
        activation="LeakyReLU",
        strides=2,
        kernel_size=4,
        norm_type=None,
        padding="same",
        name=f"C{filters}",
    )(img_input)

    # Convolution / Encoder
    for n in range(n_convBlocks):
        filters *= 2
        if n < 2:
            strides = 2
        else:
            strides = 1

        x = convBlock(
            filters=filters,
            activation="LeakyReLU",
            strides=strides,
            kernel_size=4,
            norm_type="instance",
            name=f"d{filters}",
        )(x)
    x = convBlock(
        filters=1,
        activation=None,
        padding="same",
        norm_type=None,
        strides=1,
        kernel_size=4,
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


if __name__ == "__main__":
    discriminator = Discriminator()
    tf.keras.utils.plot_model(
        discriminator,
        to_file="discriminator.png",
        show_shapes=True,
        dpi=128,
        expand_nested=True,
    )

# %%
