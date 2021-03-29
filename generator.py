# %%
import tensorflow as tf
import os
import time


def initializer(mean=0.0, stddev=0.02, seed=None):
    return tf.random_normal_initializer(mean, stddev, seed)


def addBlock(block_type: "conv/deconv", filters, kernel_size: "int | tuple | list", batchnorm: bool = True, dropout: "0 to 1" = 0):
    """
  Arguments:
    block_type: conv for Conv2D blocks, deconv for Conv2DTranspose blocks.
    filters: Integer, the dimensionality of the output space (i.e. the number of
      output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height
      and width of the 2D convolution window. Can be a single integer to specify
      the same value for all spatial dimensions.
    batchnorm: only applies to convolutional blocks.

    """
    init = initializer()

    block = tf.keras.Sequential()
    if block_type == "conv":
        block.add(
            tf.keras.layers.Conv2D(
                filters,
                kernel_size,
                strides=2,
                padding="same",
                kernel_initializer=init,
                use_bias=False)
        )
    elif block_type == "deconv":
        block.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=2,
                padding="same",
                kernel_initializer=init,
                use_bias=False)
        )
    else:
        raise NameError("block_type can only be conv or deconv")
    if batchnorm:
        block.add(tf.keras.layers.BatchNormalization())
    if dropout > 0:
        block.add(tf.keras.layers.Dropout(dropout))
    block.add(tf.keras.layers.LeakyReLU())
    return block
# %%
def createStack(block_type: "conv/deconv", input_shape: int, input_filters: int, output_shape: int, output_filters: int, kernel_size=4):

    # Conv Example
    # tensor shape: 1 > 2 > 4 > 8 > 16 > 32 > 64 > 128 > 256
    # filters shape: 768 > 384 > 192 > 96 > 48 > 3
    shape_list = []
    if block_type == "conv":
        while input_shape > output_shape:
            input_shape //= 2
            shape_list.append(input_shape)
        filters_list = [filters := input_filters * 2**4]
        while filters < output_filters:
            filters *= 2
            filters_list.append(filters)
        diff_lists = len(shape_list) - len(filters_list)
        if diff_lists > 0:
            filters_list += [filters_list[-1]]*diff_lists

    # Deconv example
    # tensor shape: 256 > 128 > 64 > 32 > 16 > 8 > 4 > 2 > 1
    # filters shape: 3 > 48 > 96 > 192 > 384 > 768
    elif block_type == "deconv":
        while input_shape < output_shape:
            input_shape *= 2
            shape_list.append(input_shape)
        filters_list = [filters := input_filters]
        while filters > output_filters:
            filters //= 2
            filters_list.append(filters)
        diff_lists = len(shape_list) - len(filters_list)
        if diff_lists > 0:
            filters_list = [filters_list[0]] * diff_lists + filters_list
    else:
        raise NameError("block_type must be either conv or deconv")

    stack = [addBlock(block_type, filters=f, kernel_size=kernel_size)
             for f in filters_list]
    print(f"{len(filters_list)} {filters_list = }")
    print(f"{len(shape_list)} {shape_list = }")
    
    return stack


print("conv:")
createStack("conv", (input_shape := 256), (input_filters := 3), (output_shape := 1), (output_filters := 384))
print("deconv:")
createStack("deconv", output_shape, output_filters*2, input_shape//2 , input_filters*2**4)
print()

# %%
def Generator(input_shape, input_filters, conv_out_shape, conv_out_filters, kernel_size=4):
    inputs = tf.keras.layers.Input(
        shape=[input_shape, input_shape, input_filters])
    conv_stack = createStack(
        "conv", input_shape=input_shape, input_filters=input_filters, output_shape=conv_out_shape, output_filters=conv_out_filters, kernel_size=kernel_size)
    deconv_stack = createStack(
        "deconv", input_shape=conv_out_shape*2, input_filters=conv_out_filters*2, output_shape=input_shape//2, output_filters=input_filters * 2**4, kernel_size=kernel_size)
    
    last_layer = tf.keras.layers.Conv2DTranspose(
        input_filters, 4, strides=2, padding="same", kernel_initializer=initializer(), activation="tanh"
    )

    # Create skip links while downsampling
    skips = []
    x = inputs
    print("conv")
    print(x.shape)
    for conv in conv_stack:
        x = conv(x)
        skips.append(x)
        print(x.shape)
        
    # remove the last layer from the skip links
    print(skips)
    skips = reversed(skips[:-1])
    print("deconv")
    # deconvolution and establish the skip connections shown in paper
    i = 0
    for deconv, skip in zip(deconv_stack, skips):
        i += 1
        x = deconv(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        print(i, x.shape)

    x = last_layer(x)
    print("last layer:", x.shape)
    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator(input_shape=256, input_filters=3, conv_out_shape=1, conv_out_filters=384)
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=200)
# %%
