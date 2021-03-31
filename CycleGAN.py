import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
from tensorflow_addons.optimizers import RectifiedAdam
from keras.losses import MeanSquaredError
import os
import time

def gen_loss_fn(fake):
    mse = MeanSquaredError()
    return mse(tf.ones_like(fake), fake)

def disc_loss_fn(real, fake):
    mse = MeanSquaredError()
    real_loss = mse(tf.ones_like(real), real)
    fake_loss = mse(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) / 2

class CycleGAN(keras.Model):
    """Args:
        generator_G: Translates X to Y
        generator_F: Translates Y to X
        discriminator_X: Discriminate F(Y) -> X
        discriminator_Y: Discriminate G(X) -> Y
        lambda_cycle: lambda value for cycle-consistency loss
        lambda_identity: lambda value for identity mapping loss 
        """

    def __init__(
        self,
        generator_G: "G(X) -> Y",
        generator_F: "F(X) -> Y",
        discriminator_X: "D_X",
        discriminator_Y: "D_Y",
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super(CycleGAN, self).__init__()
        self.G = generator_G
        self.F = generator_F
        self.D_X = discriminator_X
        self.D_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
    

    def compile(self, lr, beta1):
        """
        lr: Learning Rate of Generators for Rectified Adam
        beta1: beta1 for Rectified Adam
        """
        super(CycleGAN, self).compile()
        self.G_opt = RectifiedAdam(learning_rate=lr, beta1=beta1)
        self.F_opt = RectifiedAdam(learning_rate=lr, beta1=beta1)
        self.D_X_opt = RectifiedAdam(learning_rate=lr, beta1=beta1)
        self.D_Y_opt = RectifiedAdam(learning_rate=lr, beta1=beta1)
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_consistency_loss_fn = MeanSquaredError()
        self.identity_loss_fn = MeanSquaredError()

    def train_step(self, batch_data):
        x, y = batch_data

        # Record passes for gradient calculation
        with tf.GradientTape(persistent=True) as tape:
        
            # G(x) -> Y_hat
            Y_hat = self.G(x, training=True)
            
            # D_Y(Y_hat) -> Y_hat_D
            Y_hat_D = self.D_Y(Y_hat)
            
            # F(Y_hat) -> x_hat
            x_hat = self.F(Y_hat, training=True)
        
            # F(y) -> X_hat
            X_hat = self.F(y, training=True)
            
            # G(X_hat) -> y_hat
            y_hat = self.G(X_hat, training=True)

            # Identity mapping where F(x) -> x_i and G(y) -> y_i
            x_i = self.F(x, training=True)
            y_i = self.G(y, training=True)
