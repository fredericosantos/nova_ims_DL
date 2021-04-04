# %%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
import os
import time
from utils import ReflectionPadding2D, initializer
from blocks import convBlock


def PatchGAN(
    filters: int = 64,
    input_shape: int = 256,
    input_filters: int = 3,
    n_convBlocks: int = 3,
    name=None,
):
    """PatchGAN Discriminator for CycleGAN
    Args:
        filters: number of filters to use on the first convolution
        input_shape: image must be a square with input_shape x input_shape dimensions
        input_filters: number of channels the image has, RGB has 3 input_filters
        n_convBlocks: number of convolutional layers to go through before going through the residual block layers
        name: name of the Generator to be fed into keras
        """

    # Input layer
    img_input = tf.keras.layers.Input(
        shape=[input_shape, input_shape, input_filters], name="Input"
    )
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

    # Convolutional layers
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

    # Final Convolutional Layer that transforms the image into a  X by X with 1 feature map
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
    discriminator = PatchGAN()
    tf.keras.utils.plot_model(
        discriminator,
        to_file="architecture_PatchGAN.png",
        show_shapes=True,
        dpi=128,
        expand_nested=True,
    )

# %%
