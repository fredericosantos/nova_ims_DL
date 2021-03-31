import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
import os
import time


class CycleGAN(keras.Model):
    def __init__(self, generator_G: "G(X) -> Y", generator_F: "F(X) -> Y", discriminator_Y: "D_Y", discriminator_X: "D_X", lambda_cycle=10.0, lambda_identity=0.5):
        
