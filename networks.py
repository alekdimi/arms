import scipy.stats
import numpy as np
from absl import logging

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.stats.leave_one_out import log_soomean_exp

keras = tf.keras
tfd = tfp.distributions

EPS = 1e-6


def safe_log_prob(p):
  return tf.math.log(tf.clip_by_value(p, EPS, 1.0))


def logit_func(prob_tensor):
  """Calculate logits."""
  return safe_log_prob(prob_tensor) - safe_log_prob(1. - prob_tensor)


class BinaryNetwork(tf.keras.Model):
  """Network generating binary samples."""

  def __init__(self,
               hidden_sizes,
               activations,
               mean_xs=None,
               demean_input=False,
               final_layer_bias_initializer='zeros',
               name='binarynet'):

    super(BinaryNetwork, self).__init__(name=name)
    assert len(activations) == len(hidden_sizes)

    num_layers = len(hidden_sizes)
    self.hidden_sizes = hidden_sizes
    self.activations = activations
    self.networks = keras.Sequential()

    if demean_input:
      if mean_xs is not None:
        self.networks.add(
            tf.keras.layers.Lambda(lambda x: x - mean_xs))
      else:
        self.networks.add(
            tf.keras.layers.Lambda(lambda x: 2.*tf.cast(x, tf.float32) - 1.))
    for i in range(num_layers-1):
      self.networks.add(
          keras.layers.Dense(
              units=hidden_sizes[i],
              activation=activations[i]))

    self.networks.add(
        keras.layers.Dense(
            units=hidden_sizes[-1],
            activation=activations[-1],
            bias_initializer=final_layer_bias_initializer))

  def __call__(self,
               input_tensor,
               samples=None,
               num_samples=(),
               half_p_trick=False):
    logits = self.get_logits(input_tensor, half_p_trick)
    dist = tfd.Bernoulli(logits=logits)
    if samples is None:
      samples = dist.sample(num_samples)
    samples = tf.cast(samples, tf.float32)
    likelihood = dist.log_prob(samples)
    return samples, likelihood, logits

  def get_logits(self, input_tensor, half_p_trick=False):
    logits = self.networks(input_tensor)
    if half_p_trick:
      logits = tf.math.log_sigmoid(logits-tf.math.log(2.))
    return logits

  def sample_uniform_variables(self, sample_shape, nfold=1):
    if nfold > 1:
      sample_shape = tf.concat(
          [sample_shape[0:1] * nfold, sample_shape[1:]],
          axis=0)
    return tf.random.uniform(shape=sample_shape, maxval=1.0)


class SingleLayerDiscreteVAE(tf.keras.Model):
  """Discrete VAE as described in ARM, (Yin and Zhou (2019))."""

  def __init__(self,
               encoder,
               decoder,
               prior_logits,
               grad_type='arm',
               half_p_trick=False,
               epsilon=0.,
               control_nn=None,
               name='dvae'):
    super(SingleLayerDiscreteVAE, self).__init__(name)
    self.encoder = encoder
    self.decoder = decoder

    self.half_p_trick = half_p_trick
    if self.half_p_trick:
      # This is to enforce the p of Bernoulli distribution is [0, 0.5].
      self.prior_logits = tf.math.log_sigmoid(prior_logits-tf.math.log(2.))
    else:
      self.prior_logits = prior_logits

    self.prior_dist = tfd.Bernoulli(logits=self.prior_logits)

    self.grad_type = grad_type.lower()

    # used for variance of gradients estiamations.
    self.ema = tf.train.ExponentialMovingAverage(0.999)

    # epsilon for the tolerrence used in VIMCO-ARM++
    self.epsilon = epsilon

    if self.grad_type == 'relax':
      self.log_temperature_variable = tf.Variable(
          initial_value=tf.math.log(0.1),  # Reasonable init
          dtype=tf.float32)

      # the scaling_factor is a trainable ~1.
      self.scaling_variable = tf.Variable(
          initial_value=1.,
          dtype=tf.float32)

      # neural network for control variates lambda * r(z)
      self.control_nn = control_nn

  def call(self, input_tensor, hidden_samples=None, num_samples=()):
    """Returns ELBO.

    Args:
      input_tensor: a `float` Tensor for input observations.
        The tensor is of the shape [batch_size, observation_dims].
      hidden_samples: a discrete Tensor for hidden states.
        The tensor is of the shape [batch_size, hidden_dims].
        Default to None, in which case the hidden samples will be generated
        based on num_samples.
      num_samples: 0-D or 1-D `int` Tensor. Shape of the generated samples.

    Returns:
      elbo: the ELBO with shape [batch_size].
    """
    hidden_sample, encoder_llk, encoder_logits = self.encoder(
        input_tensor,
        samples=hidden_samples,
        num_samples=num_samples,
        half_p_trick=self.half_p_trick)

    encoder_llk = tf.reduce_sum(encoder_llk, axis=-1)
    log_pb = tf.reduce_sum(
        self.prior_dist.log_prob(hidden_sample),
        axis=-1)

    decoder_llk = tf.reduce_sum(
        self.decoder(hidden_sample, input_tensor)[1],
        axis=-1)

    elbo = decoder_llk + log_pb - encoder_llk
    encoder_logits = self.threshold_around_zero(encoder_logits)

    return elbo, hidden_sample, encoder_logits, encoder_llk

  def get_elbo(self, input_tensor, hidden_tensor):
    """Returns ELBO.

    Args:
      input_tensor: a `float` Tensor for input observations.
        The tensor is of the shape [batch_size, observation_dims].
      hidden_tensor: a discrete Tensor for hidden states.
        The tensor is of the shape [batch_size, hidden_dims].

    Returns:
      elbo: the ELBO with shape [batch_size].
    """
    elbo = self.call(input_tensor, hidden_samples=hidden_tensor)[0]
    return elbo

  def get_layer_grad_estimation(
      self, input_tensor, grad_type=None, num_samples=None):
    if grad_type is None:
      grad_type = self.grad_type

    encoder_logits = self.encoder.get_logits(input_tensor)
    sigma_phi = tf.math.sigmoid(encoder_logits)

    if grad_type == 'arm':
      u_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=1)
      b1 = tf.cast(u_noise > 1. - sigma_phi, tf.float32)
      b2 = tf.cast(u_noise < sigma_phi, tf.float32)
      f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      layer_grad = (f1 - f2) * (u_noise - 0.5)

    elif grad_type == 'disarm':
      accumulator = []
      for _ in range(num_samples):  
        u_noise = self.encoder.sample_uniform_variables(
            sample_shape=tf.shape(encoder_logits),
            nfold=1)
        sigma_abs_phi = tf.math.sigmoid(tf.math.abs(encoder_logits))
        b1 = tf.cast(u_noise > 1. - sigma_phi, tf.float32)
        b2 = tf.cast(u_noise < sigma_phi, tf.float32)
        f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
        f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
        # the factor is I(b1+b2=1) * (-1)**b2 * sigma(|phi|)
        disarm_factor = ((1. - b1) * (b2) + b1 * (1. - b2)) * (-1.)**b2
        disarm_factor *= sigma_abs_phi
        layer_gradient = 0.5 * (f1 - f2) * disarm_factor
        print(layer_gradient)
        accumulator.append(layer_gradient)
      layer_grad = tf.math.accumulate_n(accumulator) / num_samples

    elif grad_type == 'loorf':
      encoder_shape = tf.shape(encoder_logits)
      batch_size, num_logits = encoder_shape[0], encoder_shape[1]
      
      u = tf.random.uniform(shape=[num_samples, batch_size, num_logits], maxval=1.0)
      p = tf.reshape(sigma_phi, [1, batch_size, num_logits])

      b = tf.cast(u < p, tf.float32)
      b_flat = tf.reshape(b, [num_samples * batch_size, num_logits])

      tiled_input_tensor = tf.tile(tf.reshape(input_tensor, [1, batch_size, -1]), [num_samples, 1, 1,])
      flat_input_tensor = tf.reshape(tiled_input_tensor, [num_samples * batch_size, -1])
      f = tf.reshape(self.get_elbo(flat_input_tensor, b_flat), [num_samples, batch_size, 1])

      fmean = tf.reduce_mean(f, axis=0, keepdims=True)
      layer_grad = tf.reduce_mean((f - fmean) * (b - p) * num_samples / (num_samples - 1), axis=0)

    elif grad_type == 'arms':
      encoder_shape = tf.shape(encoder_logits)
      batch_size, num_logits = encoder_shape[0], encoder_shape[1]
      
      u_iid = tf.random.uniform(shape=[num_samples, batch_size, num_logits], maxval=1.0)
      p = tf.reshape(sigma_phi, [1, batch_size, num_logits])

      e = -tf.math.log(u_iid)
      d = e / tf.reduce_sum(e, axis=0, keepdims=True)
      u_copula = tf.pow(1 - d, num_samples - 1)

      p05 = tf.cast(p < 0.5, tf.float32)
      u = u_copula * p05 + (1 - u_copula) * (1 - p05)

      def bivariate(p):
        term = 2 * tf.pow(p, 1 / (num_samples - 1)) - 1
        return tf.pow(tf.maximum(term, 0), num_samples - 1)

      j1 = bivariate(p)
      j2 = 2 * p - 1 + bivariate(1 - p)
      joint = j1 * p05 + j2 * (1 - p05)
      debias = p * (1 - p) / (p - joint + 1e-6)

      b = tf.cast(u < p, tf.float32)
      b_flat = tf.reshape(b, [num_samples * batch_size, num_logits])
      tiled_input_tensor = tf.tile(tf.reshape(input_tensor, [1, batch_size, -1]), [num_samples, 1, 1,])
      flat_input_tensor = tf.reshape(tiled_input_tensor, [num_samples * batch_size, -1])
      f = tf.reshape(self.get_elbo(flat_input_tensor, b_flat), [num_samples, batch_size, 1])
      fmean = tf.reduce_mean(f, axis=0, keepdims=True)
      layer_grad = tf.reduce_mean((f - fmean) * (b - p) * num_samples / (num_samples - 1) * debias, axis=0)

    elif grad_type == 'arms_normal':
      encoder_shape = tf.shape(encoder_logits)
      batch_size, num_logits = encoder_shape[0], encoder_shape[1]
      
      p = tf.reshape(sigma_phi, [1, batch_size, num_logits])
      dim = 1 * num_samples
      cov = np.ones([dim, dim]) * 1 / (1 - dim) + np.eye(dim) * (dim / (dim - 1) + 1e-3)
      
      mvn = tfd.MultivariateNormalFullCovariance([0] * dim, cov)
      uvn = tfd.Normal(loc=0., scale=1.)
      bvn = scipy.stats.multivariate_normal(mean=[0, 0], allow_singular=True,
                                            cov=np.array([[1.001, 1 / (1 - dim)], [1 / (1 - dim), 1.001]]))
      normals = tf.cast(tf.transpose(mvn.sample(encoder_shape), perm=[2, 0, 1]), tf.float32)
      p_inv = uvn.quantile(p)

      def np_joint(p_inv):
        return bvn.cdf(np.transpose(np.concatenate([p_inv, p_inv], axis=0), axes=[1, 2, 0])).astype(np.float32)

      debias = p * (1 - p) / (1e-6 + p - tf.numpy_function(np_joint, [p_inv], tf.float32))
        
      b = tf.cast(normals < p_inv, tf.float32)  
      b_flat = tf.reshape(b, [num_samples * batch_size, num_logits])
      tiled_input_tensor = tf.tile(tf.reshape(input_tensor, [1, batch_size, -1]), [num_samples, 1, 1,])
      flat_input_tensor = tf.reshape(tiled_input_tensor, [num_samples * batch_size, -1])
      f = tf.reshape(self.get_elbo(flat_input_tensor, b_flat), [num_samples, batch_size, 1])

      fmean = tf.reduce_mean(f, axis=0, keepdims=True)
      layer_grad = tf.reduce_mean((f - fmean) * (b - p) * num_samples / (num_samples - 1) * debias, axis=0)

    else:
      raise NotImplementedError('Gradient type %s is not supported.'%grad_type)

    return layer_grad


  def sample_binaries_with_loss(
      self,
      input_tensor,
      antithetic_sample=True):
    encoder_logits = self.encoder.get_logits(input_tensor)
    sigma_phi = tf.math.sigmoid(encoder_logits)
    scaling_factor = 0.5 if self.half_p_trick else 1.
    bernoulli_prob = scaling_factor * sigma_phi
    # returned u_noise would be of the shape [batch x num_samples, event_dim].
    u_noise = self.encoder.sample_uniform_variables(
        sample_shape=tf.shape(encoder_logits))

    theresholded_encoder_logits = self.threshold_around_zero(encoder_logits)

    if antithetic_sample:
      b1 = tf.cast(u_noise > 1. - bernoulli_prob, tf.float32)
      b2 = tf.cast(u_noise < bernoulli_prob, tf.float32)
      elbo_b1 = self.get_elbo(input_tensor, b1)
      elbo_b2 = self.get_elbo(input_tensor, b2)

      return b1, b2, elbo_b1, elbo_b2, theresholded_encoder_logits

    else:
      b = tf.cast(u_noise < bernoulli_prob, tf.float32)
      elbo = self.get_elbo(input_tensor, b)
      return b, elbo, theresholded_encoder_logits


  def get_relax_parameters(
      self,
      input_tensor,
      temperature=None,
      scaling_factor=None,
      num_samples=1):
    if temperature is None:
      temperature = tf.math.exp(self.log_temperature_variable)
    if scaling_factor is None:
      scaling_factor = self.scaling_variable
    # [batch, hidden_units]
    encoder_logits = self.encoder.get_logits(input_tensor)
  
    accumulator = {'elbo': [], 'conv': [], 'conc': [], 'logq': []}
    for _ in range(num_samples):
      # returned uniform_noise would be of the shape
      # [batch x 2, event_dim].
      uniform_noise = self.encoder.sample_uniform_variables(
          sample_shape=tf.shape(encoder_logits),
          nfold=2)
      # u_noise and v_noise are both of [batch, event_dim].
      u_noise, v_noise = tf.split(uniform_noise, num_or_size_splits=2, axis=0)

      theta = tf.math.sigmoid(encoder_logits)
      z = encoder_logits + logit_func(u_noise)
      b_sample = tf.cast(z > 0, tf.float32)

      v_prime = (b_sample * (v_noise * theta + 1 - theta)
                + (1 - b_sample) * v_noise * (1 - theta))
      # z_tilde ~ p(z | b)
      z_tilde = encoder_logits + logit_func(v_prime)

      elbo = self.get_elbo(input_tensor, b_sample)
      control_variate = self.get_relax_control_variate(
          input_tensor, z,
          temperature=temperature, scaling_factor=scaling_factor)
      conditional_control = self.get_relax_control_variate(
          input_tensor, z_tilde,
          temperature=temperature, scaling_factor=scaling_factor)

      log_q = tfd.Bernoulli(logits=encoder_logits).log_prob(b_sample)

      accumulator['elbo'].append(elbo)
      accumulator['conv'].append(control_variate)
      accumulator['conc'].append(conditional_control)
      accumulator['logq'].append(log_q)

    elbos = tf.math.accumulate_n(accumulator['elbo']) / num_samples
    convs = tf.math.accumulate_n(accumulator['conv']) / num_samples
    concs = tf.math.accumulate_n(accumulator['conc']) / num_samples
    logqs = tf.math.accumulate_n(accumulator['logq']) / num_samples
    return elbos, convs, concs, logqs
    #return elbo, control_variate, conditional_control, log_q

  def get_relax_control_variate(self, input_tensor, z_sample,
                                temperature, scaling_factor):
    control_value = (
        scaling_factor *
        self.get_elbo(input_tensor, tf.math.sigmoid(z_sample/temperature)))
    if self.control_nn is not None:
      control_nn_input = tf.concat((input_tensor, z_sample), axis=-1)
      control_value += (scaling_factor
                        * tf.squeeze(self.control_nn(control_nn_input),
                                     axis=-1))
    return control_value

  def _get_grad_variance(self, grad_variable, grad_sq_variable, grad_tensor):
    grad_variable.assign(grad_tensor)
    grad_sq_variable.assign(tf.square(grad_tensor))
    self.ema.apply([grad_variable, grad_sq_variable])

    # mean per component variance
    grad_var = (
        self.ema.average(grad_sq_variable)
        - tf.square(self.ema.average(grad_variable)))
    return grad_var

  def compute_grad_variance(
      self,
      grad_variables,
      grad_sq_variables,
      grad_tensors):
    grad_var = [
        tf.reshape(self._get_grad_variance(*g), [-1])
        for g in zip(grad_variables, grad_sq_variables, grad_tensors)]
    return tf.reduce_mean(tf.concat(grad_var, axis=0))

  def threshold_around_zero(self, input_tensor):
    if self.epsilon > 0.:
      return (tf.where(tf.math.greater(input_tensor, 0.),
                       tf.math.maximum(input_tensor, self.epsilon),
                       tf.math.minimum(input_tensor, -self.epsilon)))
    return input_tensor

  @property
  def encoder_vars(self):
    return self.encoder.trainable_variables

  @property
  def decoder_vars(self):
    return self.decoder.trainable_variables

  @property
  def prior_vars(self):
    return self.prior_dist.trainable_variables


  def get_losses_arms(self, input_tensor, num_samples):
    encoder_logits = self.encoder.get_logits(input_tensor)
    sigma_phi = tf.math.sigmoid(encoder_logits)
    encoder_shape = tf.shape(encoder_logits)
    batch_size, num_logits = encoder_shape[0], encoder_shape[1]
    
    p = tf.reshape(sigma_phi, [1, batch_size, num_logits])
    p05 = tf.cast(p < 0.5, tf.float32)
    tiled_input_tensor = tf.tile(tf.reshape(input_tensor, [1, batch_size, -1]), [num_samples, 1, 1,])
    flat_input_tensor = tf.reshape(tiled_input_tensor, [num_samples * batch_size, -1])

    u  = tf.random.uniform(shape=[num_samples, batch_size, num_logits], maxval=1.0)
    b = tf.cast(u < p, tf.float32)
    b_flat = tf.reshape(b, [num_samples * batch_size, num_logits])
    log_r = tf.reshape(self.get_elbo(flat_input_tensor, b_flat), [num_samples, batch_size, 1])

    u_iid = tf.random.uniform(shape=[num_samples, batch_size, num_logits], maxval=1.0)
    e = -tf.math.log(u_iid)
    d = e / tf.reduce_sum(e, axis=0, keepdims=True)
    u_copula = tf.pow(1 - d, num_samples - 1)      
    ut = u_copula * p05 + (1 - u_copula) * (1 - p05)
    def bivariate(p):
      term = 2 * tf.pow(p, 1 / (num_samples - 1)) - 1
      return tf.pow(tf.maximum(term, 0), num_samples - 1)
    j1 = bivariate(p)
    j2 = 2 * p - 1 + bivariate(1 - p)
    joint = j1 * p05 + j2 * (1 - p05)
    debias = p * (1 - p) / (p - joint + 1e-6)


    bt = tf.cast(ut < p, tf.float32)
    bt_flat = tf.reshape(bt, [num_samples * batch_size, num_logits])
    log_rt = tf.reshape(self.get_elbo(flat_input_tensor, bt_flat), [num_samples, batch_size, 1])

    log_r_tiled = tf.tile(tf.expand_dims(log_r, axis=1), tf.constant([1, num_samples, 1, 1], tf.int32)) 

    accumulator = []
    for kth in range(num_samples):  
      mask = tf.cast(tf.ones(num_samples) - tf.one_hot(kth, num_samples) , tf.bool)
      catted = tf.concat([tf.boolean_mask(log_r_tiled, mask, axis=0), 
                                          tf.expand_dims(log_rt, axis=0)], axis=0)
      f = tf.reduce_logsumexp(catted, axis=0) - tf.math.log(num_samples * 1.0)
      fmean = tf.reduce_mean(f, axis=0, keepdims=True)
      layer_gradient = tf.reduce_mean((f - fmean) * (bt - p) * num_samples / (num_samples - 1)  * debias, axis=0)
      accumulator.append(layer_gradient)

    infnet_grad_multiplier = tf.stop_gradient(tf.math.accumulate_n(accumulator) / num_samples)
    genmo_loss = -1. * tf.squeeze(tf.reduce_logsumexp(log_r, axis=0) - tf.math.log(num_samples * 1.0))
    infnet_loss = -1. * (infnet_grad_multiplier * encoder_logits)
    return genmo_loss, infnet_loss


  def get_losses_loorf(self, input_tensor, num_samples):
    encoder_logits = self.encoder.get_logits(input_tensor)
    sigma_phi = tf.math.sigmoid(encoder_logits)
    encoder_shape = tf.shape(encoder_logits)
    batch_size, num_logits = encoder_shape[0], encoder_shape[1]
    
    p = tf.reshape(sigma_phi, [1, batch_size, num_logits])
    tiled_input_tensor = tf.tile(tf.reshape(input_tensor, [1, batch_size, -1]), [num_samples, 1, 1,])
    flat_input_tensor = tf.reshape(tiled_input_tensor, [num_samples * batch_size, -1])

    u  = tf.random.uniform(shape=[num_samples, batch_size, num_logits], maxval=1.0)
    b = tf.cast(u < p, tf.float32)
    b_flat = tf.reshape(b, [num_samples * batch_size, num_logits])
    log_r = tf.reshape(self.get_elbo(flat_input_tensor, b_flat), [num_samples, batch_size, 1])

    ut = tf.random.uniform(shape=[num_samples, batch_size, num_logits], maxval=1.0)
    bt = tf.cast(ut < p, tf.float32)
    bt_flat = tf.reshape(bt, [num_samples * batch_size, num_logits])
    log_rt = tf.reshape(self.get_elbo(flat_input_tensor, bt_flat), [num_samples, batch_size, 1])

    log_r_tiled = tf.tile(tf.expand_dims(log_r, axis=1), tf.constant([1, num_samples, 1, 1], tf.int32)) 

    accumulator = []
    for kth in range(num_samples):  
      mask = tf.cast(tf.ones(num_samples) - tf.one_hot(kth, num_samples) , tf.bool)
      catted = tf.concat([tf.boolean_mask(log_r_tiled, mask, axis=0), 
                                          tf.expand_dims(log_rt, axis=0)], axis=0)
      f = tf.reduce_logsumexp(catted, axis=0) - tf.math.log(num_samples * 1.0)
      fmean = tf.reduce_mean(f, axis=0, keepdims=True)
      layer_gradient = tf.reduce_mean((f - fmean) * (bt - p) * num_samples / (num_samples - 1), axis=0)
      accumulator.append(layer_gradient)

    infnet_grad_multiplier = tf.stop_gradient(tf.math.accumulate_n(accumulator) / num_samples)
    genmo_loss = -1. * tf.squeeze(tf.reduce_logsumexp(log_r, axis=0) - tf.math.log(num_samples * 1.0))
    infnet_loss = -1. * (infnet_grad_multiplier * encoder_logits)
    return genmo_loss, infnet_loss


  def get_relax_loss(self, input_batch, temperature=None, scaling_factor=None, num_samples=1):
    # elbo, control_variate, conditional_control should be of [batch_size]
    # log_q is of [batch_size, event_dim]
    elbo, control_variate, conditional_control, log_q = (
        self.get_relax_parameters(input_batch, temperature=temperature, scaling_factor=scaling_factor, num_samples=num_samples))
    genmo_loss = -1. * elbo
    reparam_loss = -1. * (control_variate - conditional_control)

    # [batch_size]
    learning_signal = -1. * (elbo - conditional_control)
    self.mean_learning_signal = tf.reduce_mean(learning_signal)

    # [batch_size, hidden_size]
    learning_signal = tf.tile(
        tf.expand_dims(learning_signal, axis=-1),
        [1, tf.shape(log_q)[-1]])

    return genmo_loss, reparam_loss, learning_signal, log_q
