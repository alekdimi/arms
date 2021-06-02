import os

from absl import app, flags

import dataset
import networks
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
layers = tf.keras.layers

flags.DEFINE_enum('dataset', 'static_mnist', ['static_mnist', 'dynamic_mnist', 'fashion_mnist', 'omniglot'], 'Dataset to use.')
flags.DEFINE_float('genmo_lr', 1e-4, 'Learning rate for decoder, Generation network.')
flags.DEFINE_float('infnet_lr', 1e-4, 'Learning rate for encoder, Inference network.')
flags.DEFINE_float('prior_lr', 1e-2, 'Learning rate for prior variables.')
flags.DEFINE_integer('batch_size', 50, 'Training batch size.')
flags.DEFINE_integer('num_pairs', 1, ('Number of sample pairs used gradient estimators.'))
flags.DEFINE_integer('num_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_string('encoder_type', 'linear', 'Choice supported: linear, nonlinear')
flags.DEFINE_string('grad_type', 'arm', 'Choice supported: arm, disarm, reinforce')
flags.DEFINE_string('logdir', 'logs/tmp', 'Directory for storing logs.')
flags.DEFINE_bool('verbose', False, 'Whether to turn on training result logging.')
flags.DEFINE_integer('repeat_idx', 0, 'Dummy flag to label the experiments in repeats.')
flags.DEFINE_bool('half_p_trick', False, 'Enforce the p range is [0., 0.5]')
flags.DEFINE_float('epsilon', 0., 'Additive float to prevent numerical underflow in log(x).')
flags.DEFINE_float('temperature', None, 'Temperature for RELAX estimator.')
flags.DEFINE_float('scaling_factor', None, 'Scaling factor for RELAX estimator.')
flags.DEFINE_bool('eager', False, 'Enable eager execution.')
flags.DEFINE_bool('bias_check', False, 'Carry out bias check for RELAX and baseline')
flags.DEFINE_bool('demean_input', False, 'Demean for encoder and decoder inputs.')
flags.DEFINE_bool('initialize_with_bias', False, 'Initialize the final layer bias of decoder with dataset mean.')
flags.DEFINE_integer('seed', 1, 'Global random seed.')
flags.DEFINE_integer('num_eval_samples', None, 'Number of samples for evaluation, default to num_pairs.')
flags.DEFINE_integer('num_train_samples', None, 'Number of samples for evaluation, default to num_pairs.')
flags.DEFINE_bool('debug', False, 'Turn on debugging mode.')
FLAGS = flags.FLAGS


def process_batch_input(input_batch):
  input_batch = tf.reshape(input_batch, [tf.shape(input_batch)[0], -1])
  input_batch = tf.cast(input_batch, tf.float32)
  return input_batch


def initialize_grad_variables(target_variable_list):
  return [tf.Variable(tf.zeros(shape=i.shape)) for i in target_variable_list]


def estimate_gradients(input_batch, bvae_model, gradient_type, sample_size=1):
  if gradient_type == 'relax':
    with tf.GradientTape(persistent=True) as tape:
      genmo_loss, reparam_loss, learning_signal, log_q = (
          bvae_model.get_relax_loss(input_batch, temperature=FLAGS.temperature,
          scaling_factor=FLAGS.scaling_factor, num_samples=sample_size))

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    infnet_vars = bvae_model.encoder_vars
    infnet_grads_1 = tape.gradient(log_q, infnet_vars, output_gradients=learning_signal)
    infnet_grads_2 = tape.gradient(reparam_loss, infnet_vars)
    infnet_grads = [infnet_grads_1[i] + infnet_grads_2[i] for i in range(len(infnet_vars))]

  else:
    with tf.GradientTape(persistent=True) as tape:
      elbo, _, infnet_logits, _ = bvae_model(input_batch)
      genmo_loss = -1. * tf.reduce_mean(elbo)

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    infnet_grad_multiplier = -1. * bvae_model.get_layer_grad_estimation(input_batch, num_samples=sample_size)
    infnet_grads = tape.gradient(infnet_logits, bvae_model.encoder_vars, output_gradients=infnet_grad_multiplier)

  del tape
  return (genmo_grads, prior_grads, infnet_grads, genmo_loss)


@tf.function
def train_one_step(
    train_batch_i,
    bvae_model,
    genmo_optimizer,
    infnet_optimizer,
    prior_optimizer,
    theta_optimizer,
    encoder_grad_variable,
    encoder_grad_sq_variable):
  """Train Discrete VAE for 1 step."""
  metrics = {}
  input_batch = process_batch_input(train_batch_i)
  if FLAGS.grad_type in ['loorf', 'arms', 'arms_normal']:
    num_samples = 2 * FLAGS.num_pairs
  else:
    num_samples = FLAGS.num_pairs

  if FLAGS.grad_type == 'relax':
    with tf.GradientTape(persistent=True) as theta_tape:
      (genmo_grads, prior_grads, infnet_grads, genmo_loss) = estimate_gradients(
          input_batch, bvae_model, FLAGS.grad_type, num_samples)

      genmo_vars = bvae_model.decoder_vars
      genmo_optimizer.apply_gradients(list(zip(genmo_grads, genmo_vars)))

      prior_vars = bvae_model.prior_vars
      prior_optimizer.apply_gradients(list(zip(prior_grads, prior_vars)))

      infnet_vars = bvae_model.encoder_vars
      infnet_optimizer.apply_gradients(list(zip(infnet_grads, infnet_vars)))

      infnet_grads_sq = [tf.square(grad_i) for grad_i in infnet_grads]
      theta_vars = []
      if bvae_model.control_nn:
        theta_vars.extend(bvae_model.control_nn.trainable_variables)
      if FLAGS.temperature is None:
        theta_vars.append(bvae_model.log_temperature_variable)
      if FLAGS.scaling_factor is None:
        theta_vars.append(bvae_model.scaling_variable)
      theta_grads = theta_tape.gradient(infnet_grads_sq, theta_vars)
      theta_optimizer.apply_gradients(zip(theta_grads, theta_vars))
    del theta_tape

    metrics['learning_signal'] = bvae_model.mean_learning_signal

  else:
    (genmo_grads, prior_grads, infnet_grads, genmo_loss) = estimate_gradients(
        input_batch, bvae_model, FLAGS.grad_type, num_samples)

    genmo_vars = bvae_model.decoder_vars
    genmo_optimizer.apply_gradients(list(zip(genmo_grads, genmo_vars)))

    prior_vars = bvae_model.prior_vars
    prior_optimizer.apply_gradients(list(zip(prior_grads, prior_vars)))

    infnet_vars = bvae_model.encoder_vars
    infnet_optimizer.apply_gradients(list(zip(infnet_grads, infnet_vars)))

  batch_size_sq = tf.cast(FLAGS.batch_size * FLAGS.batch_size, tf.float32)
  encoder_grad_var = bvae_model.compute_grad_variance(
      encoder_grad_variable, encoder_grad_sq_variable,
      infnet_grads) / batch_size_sq

  return (encoder_grad_var, None, genmo_loss, metrics)


@tf.function
def evaluate(model, tf_dataset, max_step=1000, num_eval_samples=None):
  """Evaluate the model."""
  if num_eval_samples:
    num_samples = num_eval_samples
  elif FLAGS.num_eval_samples:
    num_samples = FLAGS.num_eval_samples
  elif FLAGS.grad_type in ['vimco', 'local-disarm', 'local-arms']:
    num_samples = FLAGS.num_pairs * 2
  elif FLAGS.grad_type in ['loorf', 'arms', 'arms_normal']:
    num_samples = 2 * FLAGS.num_pairs
  else:
    num_samples = FLAGS.num_pairs
  loss = 0.
  n = 0.
  for batch in tf_dataset.map(process_batch_input):
    if n >= max_step:  # used for train_ds, which is a `repeat` dataset.
      break
    if num_samples > 1:
      batch_size = tf.shape(batch)[0]
      input_batch = tf.tile(batch, [num_samples, 1])
      elbo = tf.reshape(model(input_batch)[0], [num_samples, batch_size])
      objectives = (tf.reduce_logsumexp(elbo, axis=0, keepdims=False) -
                    tf.math.log(tf.cast(tf.shape(elbo)[0], tf.float32)))
    else:
      objectives = model(batch)[0]
    loss -= tf.reduce_mean(objectives)
    n += 1.
  return loss / n


def main(_):

  tf.random.set_seed(FLAGS.seed)
  logdir = FLAGS.logdir
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  if FLAGS.eager:
    tf.config.experimental_run_functions_eagerly(FLAGS.eager)

  genmo_lr = tf.constant(FLAGS.genmo_lr)
  infnet_lr = tf.constant(FLAGS.infnet_lr)
  prior_lr = tf.constant(FLAGS.prior_lr)

  genmo_optimizer = tf.keras.optimizers.Adam(learning_rate=genmo_lr)
  infnet_optimizer = tf.keras.optimizers.Adam(learning_rate=infnet_lr)
  prior_optimizer = tf.keras.optimizers.SGD(learning_rate=prior_lr)
  theta_optimizer = tf.keras.optimizers.Adam(learning_rate=infnet_lr,
                                             beta_1=0.999)

  batch_size = FLAGS.batch_size

  if FLAGS.dataset == 'static_mnist':
    train_ds, valid_ds, test_ds = dataset.get_static_mnist_batch(batch_size)
    train_size = 50000
  elif FLAGS.dataset == 'dynamic_mnist':
    train_ds, valid_ds, test_ds = dataset.get_dynamic_mnist_batch(batch_size)
    train_size = 50000
  elif FLAGS.dataset == 'fashion_mnist':
    train_ds, valid_ds, test_ds = dataset.get_dynamic_mnist_batch(
        batch_size, fashion_mnist=True)
    train_size = 50000
  elif FLAGS.dataset == 'omniglot':
    train_ds, valid_ds, test_ds = dataset.get_omniglot_batch(batch_size)
    train_size = 23000

  num_steps_per_epoch = int(train_size / batch_size)
  train_ds_mean = dataset.get_mean_from_iterator(
      train_ds, dataset_size=train_size, batch_size=batch_size)

  if FLAGS.initialize_with_bias:
    bias_value = -tf.math.log(
        1./tf.clip_by_value(train_ds_mean, 0.001, 0.999) - 1.).numpy()
    bias_initializer = tf.keras.initializers.Constant(bias_value)
  else:
    bias_initializer = 'zeros'

  if FLAGS.encoder_type == 'linear':
    encoder_hidden_sizes = [200]
    encoder_activations = ['linear']
    decoder_hidden_sizes = [784]
    decoder_activations = ['linear']
  elif FLAGS.encoder_type == 'nonlinear':
    encoder_hidden_sizes = [200, 200, 200]
    encoder_activations = [
        layers.LeakyReLU(alpha=0.3),
        layers.LeakyReLU(alpha=0.3),
        'linear']
    decoder_hidden_sizes = [200, 200, 784]
    decoder_activations = [
        layers.LeakyReLU(alpha=0.3),
        layers.LeakyReLU(alpha=0.3),
        'linear']
  else:
    raise NotImplementedError

  encoder = networks.BinaryNetwork(
      encoder_hidden_sizes,
      encoder_activations,
      mean_xs=train_ds_mean,
      demean_input=FLAGS.demean_input,
      name='bvae_encoder')
  decoder = networks.BinaryNetwork(
      decoder_hidden_sizes,
      decoder_activations,
      demean_input=FLAGS.demean_input,
      final_layer_bias_initializer=bias_initializer,
      name='bvae_decoder')

  prior_logit = tf.Variable(tf.zeros([200], tf.float32))

  if FLAGS.grad_type == 'relax':
    control_network = tf.keras.Sequential()
    control_network.add(
        layers.Dense(137, activation=layers.LeakyReLU(alpha=0.3)))
    control_network.add(
        layers.Dense(1))
  else:
    control_network = None

  bvae_model = networks.SingleLayerDiscreteVAE(
      encoder,
      decoder,
      prior_logit,
      grad_type=FLAGS.grad_type,
      half_p_trick=FLAGS.half_p_trick,
      epsilon=FLAGS.epsilon,
      control_nn=control_network)

  bvae_model.build(input_shape=(None, 784))

  tensorboard_file_writer = tf.summary.create_file_writer(logdir)

  encoder_grad_variable = initialize_grad_variables(bvae_model.encoder_vars)
  encoder_grad_sq_variable = initialize_grad_variables(bvae_model.encoder_vars)

  start_step = infnet_optimizer.iterations.numpy()

  train_iter = train_ds.__iter__()
  for step_i in range(start_step, FLAGS.num_steps):
    (encoder_grad_var, variance_dict, genmo_loss, metrics) = train_one_step(
        train_iter.next(),
        bvae_model,
        genmo_optimizer,
        infnet_optimizer,
        prior_optimizer,
        theta_optimizer,
        encoder_grad_variable,
        encoder_grad_sq_variable)
    train_loss = tf.reduce_mean(genmo_loss)

    if step_i % 1000 == 0:
      metrics.update({
          'train_objective': train_loss,
          'eval_metric/train': evaluate(bvae_model, train_ds, max_step=num_steps_per_epoch, num_eval_samples=FLAGS.num_train_samples),
          'eval_metric/valid': evaluate(bvae_model, valid_ds, num_eval_samples=FLAGS.num_eval_samples),
          'eval_metric/test': evaluate(bvae_model, test_ds, num_eval_samples=FLAGS.num_eval_samples),
          'var/grad': encoder_grad_var
      })
      if FLAGS.grad_type == 'relax':
        if FLAGS.temperature is None:
          metrics['relax/temperature'] = tf.math.exp(bvae_model.log_temperature_variable)
        if FLAGS.scaling_factor is None:
          metrics['relax/scaling'] = bvae_model.scaling_variable
      tf.print(step_i, metrics)

      with tensorboard_file_writer.as_default():
        for k, v in metrics.items():
          tf.summary.scalar(k, v, step=step_i)
        if variance_dict is not None:
          tf.print(variance_dict)
          for k, v in variance_dict.items():
            tf.summary.scalar(k, v, step=step_i)

if __name__ == '__main__':
  app.run(main)
