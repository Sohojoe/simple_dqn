from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, Flatten, Dense, Activation, Input, Lambda, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam, RMSprop, Adadelta
from keras import backend as K
import numpy as np

import os
import logging
logger = logging.getLogger(__name__)

class DeepQNetwork:
  def __init__(self, num_actions, args):
    # remember parameters
    self.num_actions = num_actions
    self.batch_size = args.batch_size
    self.discount_rate = args.discount_rate
    self.history_length = args.history_length
    self.screen_dim = (args.screen_height, args.screen_width)
    self.clip_error = args.clip_error
    self.min_reward = args.min_reward
    self.max_reward = args.max_reward
    self.batch_norm = args.batch_norm

    # # create Neon backend
    # self.be = gen_backend(backend = args.backend,
    #              batch_size = args.batch_size,
    #              rng_seed = args.random_seed,
    #              device_id = args.device_id,
    #              datatype = np.dtype(args.datatype).type,
    #              stochastic_round = args.stochastic_round)

    # create Keras backend
    self.k = K
    np.random.seed(args.random_seed)


    # prepare tensors once and reuse them
    self.input_shape = (self.history_length,) + self.screen_dim + (self.batch_size,)
    # self.input = self.be.empty(self.input_shape)
    # self.input.lshape = self.input_shape # HACK: needed for convolutional networks
    # self.targets = self.be.empty((self.num_actions, self.batch_size))

    # create model
    S, V = self._createLayers(num_actions, self.input_shape[:-1])
    self.model = Model(S, V)
    self.model.summary()
    self.cost = 'mse'  #TODO check is equal to 'GeneralizedCost(costfunc = SumSquared())'
    # # Bug fix
    # for l in self.model.layers.layers:
    #   l.parallelism = 'Disabled'
    if args.optimizer == 'rmsprop':
      self.optimizer = RMSprop(lr=args.learning_rate,
                               rho=args.decay_rate) # TODO check this is ok?
                               # stochastic_round=args.stochastic_round)
    elif args.optimizer == 'adam':
      self.optimizer = Adam(lr = args.learning_rate,
          stochastic_round = args.stochastic_round)
    elif args.optimizer == 'adadelta':
      self.optimizer = Adadelta(decay = args.decay_rate,
          stochastic_round = args.stochastic_round)
      assert False, "invalid optimizer"
    self.model.compile(optimizer=self.optimizer, loss=self.cost)

    # create target model
    self.target_steps = args.target_steps
    self.train_iterations = 0
    if self.target_steps:
      S, V = self._createLayers(num_actions, self.input_shape[:-1])
      self.target_model = Model(S, V)
      # Bug fix
      # for l in self.target_model.layers.layers:
      #   l.parallelism = 'Disabled'
      # self.target_model.initialize(self.input_shape[:-1])
      self.save_weights_prefix = args.save_weights_prefix
    else:
      self.target_model = self.model

    self.callback = None

  def _createLayers(self, num_actions, input_shape):
    #TODO check is right Gaussian
    # init_norm = 'glorot_normal'
    init_norm = 'glorot_uniform'
    S = Input(shape=input_shape)
    # , , self.cost
    h = Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', activation='relu', init=init_norm)(S)
    h = Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', activation='relu', init=init_norm)(h)
    h = Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu', init=init_norm)(h)
    h = Flatten()(h)
    #TODO DDQN split here--
    # Value = Dense 512 -> Dense 1
    # Advantage = Dense 512 -> Dense n.actions
    h = Dense(512, activation='relu', init=init_norm)(h)
    V = Dense(num_actions, init=init_norm)(h)
    return S,V

  def _setInput(self, states):
    # # change order of axes to match what Neon expects
    # states = np.transpose(states, axes = (1, 2, 3, 0))
    # states = np.transpose(states, axes = (0, 2, 3, 1))
    # # copy() shouldn't be necessary here, but Neon doesn't work otherwise
    # self.input.set(states.copy())
    # # normalize network input between 0 and 1
    # self.be.divide(self.input, 255, self.input)
    self.input = states

    return

  def train(self, minibatch, epoch):
    # expand components of minibatch
    prestates, actions, rewards, poststates, terminals = minibatch
    assert len(prestates.shape) == 4
    assert len(poststates.shape) == 4
    assert len(actions.shape) == 1
    assert len(rewards.shape) == 1
    assert len(terminals.shape) == 1
    assert prestates.shape == poststates.shape
    assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

    if self.target_steps and self.train_iterations % self.target_steps == 0:
      # have to serialize also states for batch normalization to work
      # pdict = self.model.get_description(get_weights=True, keep_states=True)
      # self.target_model.deserialize(pdict, load_states=True)
      weights = self.model.get_weights()
      self.target_model.set_weights(weights)

    # feed-forward pass for poststates to get Q-values
    self._setInput(poststates)
    postq = self.target_model.predict(self.input).T
    assert postq.shape == (self.num_actions, self.batch_size)

    # calculate max Q-value for each poststate
    maxpostq = np.max(postq, axis=0)
    # assert maxpostq.shape == (1, self.batch_size)

    # feed-forward pass for prestates
    self._setInput(prestates)
    preq = self.model.predict(self.input).T
    assert preq.shape == (self.num_actions, self.batch_size)

    # make copy of prestate Q-values as targets
    targets = preq

    # clip rewards between -1 and 1
    # rewards = np.clip(rewards, self.min_reward, self.max_reward)

    # update Q-value targets for actions taken
    # -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    targets = rewards + np.multiply(1-terminals, self.discount_rate * maxpostq)
    preq[0,:] = targets
    # for i, action in enumerate(actions):
    #   if terminals[i]:
    #     targets[action, i] = float(rewards[i])
    #   else:
    #     targets[action, i] = float(rewards[i]) + self.discount_rate * maxpostq[i]

    self.model.train_on_batch(self.input, preq.T)
    xx = self.model.loss_functions.

    # # copy targets to GPU memory
    # self.targets.set(targets)
    #
    # # calculate errors
    # deltas = self.cost.get_errors(preq, self.targets)
    # assert deltas.shape == (self.num_actions, self.batch_size)
    # #assert np.count_nonzero(deltas.asnumpyarray()) == 32
    #
    # # calculate cost, just in case
    # cost = self.cost.get_cost(preq, self.targets)
    # assert cost.shape == (1,1)
    #
    # # clip errors
    # if self.clip_error:
    #   self.be.clip(deltas, -self.clip_error, self.clip_error, out = deltas)
    #
    # # perform back-propagation of gradients
    # self.model.bprop(deltas)
    #
    # # perform optimization
    # self.optimizer.optimize(self.model.layers_to_optimize, epoch)

    # increase number of weight updates (needed for target clone interval)
    self.train_iterations += 1

    # calculate statistics
    # if self.callback:
    #   self.callback.on_train(cost.asnumpyarray()[0,0])

  def predict(self, states):
    # minibatch is full size, because Neon doesn't let change the minibatch size
    assert states.shape == ((self.batch_size, self.history_length,) + self.screen_dim)
    # calculate Q-values for the states
    self._setInput(states)
    qvalues = self.model.predict(self.input)
    # assert qvalues.shape == (self.num_actions, self.batch_size)
    assert qvalues.shape == (self.batch_size, self.num_actions)
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug("Q-values: " + str(qvalues.asnumpyarray()[:, 0]))
    # transpose the result, so that batch size is first dimension
    return qvalues

  def load_weights(self, load_path):
    self.model.load_weights(load_path)

  def save_weights(self, save_path):
    self.model.save_weights(save_path)
