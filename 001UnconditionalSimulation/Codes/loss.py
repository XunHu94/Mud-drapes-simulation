import numpy as np
import tensorflow.compat.v1 as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]
#----------------------------------------------------------------------------
# Generator loss function.

def G_wgan_acgan(G, D, lod, minibatch_size, orig_weight=1, batch_multiplier=1, lossnorm=False):
    # lossnorm: True to normalize loss into standard Gaussian before multiplying with weights.
    
    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out
    if lossnorm: loss = (loss -14.6829250772099) / 4.83122039859412   #To Normalize
    loss = tfutil.autosummary('Loss_G/GANloss', loss)
    loss = loss * orig_weight
     
    loss = tfutil.autosummary('Loss_G/Total_loss', loss)    
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_wgangp_acgan(G, D, opt, minibatch_size, reals,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    batch_multiplier = 1):       

    latents = tf.random_normal([minibatch_size * batch_multiplier] + G.input_shapes[0][1:])

    fake_images_out = G.get_output_for(latents, is_training=True)

    reals = tf.reshape(tf.tile(tf.expand_dims(reals, 1), [1, batch_multiplier, 1, 1, 1]),([-1] + [1] + [64] + [64]))
    real_scores_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss_D/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss_D/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size * batch_multiplier, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    loss = tfutil.autosummary('Loss_D/WGAN_GP_loss', loss)
   
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss_D/epsilon_penalty', tf.square(real_scores_out))
        loss += epsilon_penalty * wgan_epsilon

    return loss