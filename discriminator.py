# %%
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os
import time
from utils import ReflectionPadding2D, initializer
from blocks import *

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, MaxPool2D
import os
import time
from utils import ReflectionPadding2D, initializer
from blocks import *

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
        shape=[input_shape, input_shape, input_filters], name="Input")

    # Because the kernel size is 7x7 on the first convolution, we add reflective padding 3x3
    x = ReflectionPadding2D(padding=(3, 3), name="REF_0")(img_input)

    # First 7x7, convolution c7s1_64
    x = convBlock(filters=filters, activation="relu", strides=1,
                  kernel_size=7, norm_type="instance", padding="valid",
                  name=f"c7s1-{filters}")(x)

    # Convolution / Encoder
    for _ in range(n_convBlocks):
        filters *= 2
        x = convBlock(filters=filters, activation="relu",
                      norm_type="instance", name=f"d{filters}")(x)
        # MaxPooling after every 3 convolution blocks
        if ( _ % 3 == 1):
            x = MaxPool2D((2, 2), strides=2)(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Output
    x = Dense(1, activation="sigmoid")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model

discriminator = Discriminator()

tf.keras.utils.plot_model(discriminator, to_file="discriminator.png", show_shapes=True, dpi=128)
with open('discriminator.txt', 'w') as f:
    discriminator.summary(print_fn=lambda x: f.write(x + '\n'))
# %%