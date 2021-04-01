import tensorflow as tf
import tensorflow.keras as keras
from tensorflow_addons.layers import InstanceNormalization
from tensorflow_addons.optimizers import RectifiedAdam
from keras.losses import MeanSquaredError, MeanAbsoluteError
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


def disc_loss_fn_experimental(real, fake):
    mse = MeanSquaredError()
    real_loss = mse(tf.ones_like(real), real)
    fake_loss = mse(tf.ones_like(fake) / 2, fake)
    return (real_loss + fake_loss) / 2


class CycleGAN(keras.Model):
    """Args:
        generator_G: Translates X to Y
        generator_F: Translates Y to X
        discriminator_X: Discriminate F(Y) -> X
        discriminator_Y: Discriminate G(X) -> Y
        lambda_cycle: lambda value for cycle-consistency loss
        lambda_identity: lambda value for identity mapping loss
        paper_generator_loss: in the paper and code from paper,
            authors have one generator loss for both F and G.
            Set True to use the same gen loss for F and G, set
            False to use split gen losses.
        """

    def __init__(
        self,
        generator_G: "G(X) -> Y",
        generator_F: "F(X) -> Y",
        discriminator_X: "D_x",
        discriminator_Y: "D_y",
        lambda_cycle=10.0,
        lambda_identity=0.5,
        paper_generator_loss: bool = True,
    ):
        super(CycleGAN, self).__init__()
        self.G = generator_G
        self.F = generator_F
        self.D_x = discriminator_X
        self.D_y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.paper_generator_loss = paper_generator_loss

    def compile(self, lr, beta_1):
        """
        lr: Learning Rate of Generators for Rectified Adam
        beta_1: beta_1 for Rectified Adam
        """
        super(CycleGAN, self).compile()
        self.G_opt = RectifiedAdam(learning_rate=lr, beta_1=beta_1)
        self.F_opt = RectifiedAdam(learning_rate=lr, beta_1=beta_1)
        self.D_x_opt = RectifiedAdam(learning_rate=lr, beta_1=beta_1)
        self.D_y_opt = RectifiedAdam(learning_rate=lr, beta_1=beta_1)
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.discriminator_loss_fn_exp = disc_loss_fn_experimental
        self.cycle_consistency_loss_fn = MeanAbsoluteError()
        self.identity_loss_fn = MeanAbsoluteError()

    def train_step(self, batch_data):
        x, y = batch_data

        # Record passes for gradient calculation
        with tf.GradientTape(persistent=True) as tape:
            # D_y(y) -> y_D
            y_D = self.D_y(y, training=True)
            # D_x(x) -> x_D
            x_D = self.D_x(x, training=True)

            # G(x) -> Y_hat
            Y_hat = self.G(x, training=True)

            # D_y(Y_hat) -> Y_hat_D
            Y_hat_D = self.D_y(Y_hat, training=True)

            # F(Y_hat) -> x_hat
            x_hat = self.F(Y_hat, training=True)

            # F(y) -> X_hat
            X_hat = self.F(y, training=True)

            # D_x(X_hat) -> X_hat_D
            X_hat_D = self.D_x(X_hat, training=True)

            # G(X_hat) -> y_hat
            y_hat = self.G(X_hat, training=True)

            # Identity mapping where F(x) -> x_i and G(y) -> y_i
            x_i = self.F(x, training=True)
            y_i = self.G(y, training=True)

            # Generators' adversarial loss
            G_loss = self.generator_loss_fn(Y_hat_D)
            F_loss = self.generator_loss_fn(X_hat_D)

            # Generators' cycle consistency loss
            forward_cycle_loss = (
                self.cycle_consistency_loss_fn(x, x_hat) * self.lambda_cycle
            )
            backward_cycle_loss = (
                self.cycle_consistency_loss_fn(y, y_hat) * self.lambda_cycle
            )

            # Generator Identity Loss
            G_identity_loss = (
                self.identity_loss_fn(y, y_i) * self.lambda_cycle * self.lambda_identity
            )
            F_identity_loss = (
                self.identity_loss_fn(x, x_i) * self.lambda_cycle * self.lambda_identity
            )

            # Total Generator Loss as per paper
            if self.paper_generator_loss:
                generator_loss = (
                    G_loss
                    + F_loss
                    + G_identity_loss
                    + F_identity_loss
                    + forward_cycle_loss
                    + backward_cycle_loss
                )
            else:
                # Split generator loss
                G_loss_total = G_loss + G_identity_loss + F_identity_loss + forward_cycle_loss + backward_cycle_loss
                F_loss_total = F_loss + G_identity_loss + F_identity_loss + forward_cycle_loss + backward_cycle_loss


            # Discriminators losses
            D_y_loss = (
                self.discriminator_loss_fn(y_D, Y_hat_D)
            )

            D_x_loss = (
                self.discriminator_loss_fn(x_D, X_hat_D)
            )

        # Gradients for generators
        if self.paper_generator_loss:
            gradient_G = tape.gradient(generator_loss, self.G.trainable_variables)
            gradient_F = tape.gradient(generator_loss, self.F.trainable_variables)
        else:
            gradient_G = tape.gradient(G_loss_total, self.G.trainable_variables)
            gradient_F = tape.gradient(F_loss_total, self.F.trainable_variables)

        # Gradients for discriminators
        gradient_D_x = tape.gradient(D_x_loss, self.D_x.trainable_variables)
        gradient_D_y = tape.gradient(D_y_loss, self.D_y.trainable_variables)

        # Backpropagation
        self.G_opt.apply_gradients(zip(gradient_G, self.G.trainable_variables))
        self.F_opt.apply_gradients(zip(gradient_F, self.F.trainable_variables))
        self.D_x_opt.apply_gradients(zip(gradient_D_x, self.D_x.trainable_variables))
        self.D_y_opt.apply_gradients(zip(gradient_D_y, self.D_y.trainable_variables))

        return {
            "G_loss": G_loss,
            "F_loss": F_loss,
            "D_y_loss": D_y_loss,
            "D_x_loss": D_x_loss,
        }
