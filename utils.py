import tensorflow as tf
from tensorflow import keras
import os

IMG_HEIGHT = 64
IMG_WIDTH = IMG_HEIGHT
N_CHANNELS = 3
BATCH_SIZE = 1

# generates a keras.dataset object for train or val.
def train_ds_gen(dir_name, val_split=0.2, val=False, batch_size=1):
    """ Generates a dataset. 
    Returns a tf.data.Dataset object.
    Parameters: directory: the folder where the images are.
    val_split = number to feed to keras.preprocess.image from directory. Default is 0.2.
    val = whether the func generates a training or val dataset. Default generates train.
    """
    directory = os.path.join(os.getcwd(), "datasets", dir_name)

    if val:
        subset = "validation"
    else:
        subset = "training"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=directory,
        validation_split=val_split,
        subset=subset,
        seed=42,
        batch_size=BATCH_SIZE,
    )

    return train_ds


## Preprocess
## Normalize still isn't working, need to understand what, how and why
def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_img(tensor_img, tensor_label):
    tensor_img = tf.image.resize(tensor_img, [150, 150])
    tensor_img = tf.image.random_crop(tensor_img, [batch_size, 128, 128, 3])
    tensor_img = tf.image.random_flip_left_right(tensor_img)
    # tensor_img = normalize_img(img)
    return tensor_img, tensor_label


# Keeping backward compatibility
# randomly crop the image
def random_crop(image):
    return tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, N_CHANNELS])


def normalize(image):
    image = tf.cast(image, tf.float32)
    image /= 127.5
    return image - 1


def image_resize(image):
    image = tf.image.resize(
        image,
        [int(IMG_HEIGHT), int(IMG_WIDTH)],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return image


def random_jitter(image, resize_zoom: float = 1.1):
    """
    resize_zoom => Increase the image size (that will go into the network)
    by resize_zoom (eg. 255 * 1.1) before randomly cropping it by the image size (eg. 255x255).
    """

    # Resize image if we're choosing to resize
    image = tf.image.resize(
        image,
        [int(IMG_HEIGHT * resize_zoom), int(IMG_WIDTH * resize_zoom)],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    # Crop the image with the given height and width
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_image_train(image, label):

    image = random_jitter(image)
    image = normalize(image)
    return image


def preprocess_image_test(image, label):
    image = normalize(image_resize(image))
    return image


## Padding
class ReflectionPadding2D(tf.keras.layers.Layer):
    """Reflective padding used in the paper.
    https://arxiv.org/pdf/1703.10593.pdf ==> 7.2 Network architectures. 
    'Reflection padding was used to reduce artifacts.'

    Code adapted from tensorflow tutorials.
    Args:
        padding(tuple): padding for spatial dimensions
    Returns:
        Padded Tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def initializer(mean=0.0, stddev=0.02, seed=None):
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
