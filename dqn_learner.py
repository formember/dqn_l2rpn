# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""D4PG agent implementation."""

import copy
from acme.tf import savers as tf2_savers
from acme import datasets
from acme.agents.tf.dqn import learning
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf
from utils import create_variables
from acme import types

tf.config.run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class DQN_learner(object):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 policy_network: snt.Module,
                 batch_size: int = 256,
                 port: int = 8000,
                 prefetch_size: int = 4,
                 target_update_period: int = 100,
                 replay_table_max_times_sampled: float = 32.0,
                 min_replay_size: int = 1000,
                 max_replay_size: int = 1000000,
                 importance_sampling_exponent: float = 0.2,
                 priority_exponent: float = 0.6,
                 n_step: int = 5,
                 epsilon: tf.Tensor = None,
                 learning_rate: float = 1e-3,
                 discount: float = 0.95,
                 logger: loggers.Logger = None,
                 checkpoint: bool = True,
                 checkpoint_subpath: str = '~/acme/',
                 model_table_name: str = "model_table_name",
                 replay_table_name: str = "replay_table_name",
                 shutdown_table_name: str = "shutdown_table_name",
                 device_placement: str = "CPU",
                 broadcaster_table_name: str = "broadcast_table_name", ):
        """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      policy_network: the online (optimized) policy.
      critic_network: the online critic.
      observation_network: optional network to transform the observations before
        they are fed into any network.
      discount: discount to use for TD updates.
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      max_replay_size: maximum replay size.
      n_step: number of steps to squash into a single transition.
      clipping: whether to clip gradients by global norm.
      logger: logger object to be used by learner.
      counter: counter object used to keep track of steps.
      checkpoint: boolean indicating whether to checkpoint the learner.
      replay_table_name: string indicating what name to give the replay table.
    """

        transition_spec = types.Transition(
            observation=tf.TensorSpec(shape=(observation_spec,),
                              dtype=tf.float32),
            action=tf.TensorSpec(shape=(),
                              dtype=tf.int32),
            reward=tf.TensorSpec(shape=(),
                              dtype=tf.float32),
            discount=tf.TensorSpec(shape=(),
                              dtype=tf.float32),
            next_observation=tf.TensorSpec(shape=(observation_spec,),
                              dtype=tf.float32)
        )
        replay_table = reverb.Table(
            name=replay_table_name,
            sampler=reverb.selectors.Prioritized(priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            max_times_sampled=replay_table_max_times_sampled,
            rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
            signature=transition_spec
        )
        model_table = reverb.Table(
            name=model_table_name,
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=1,
            rate_limiter=reverb.rate_limiters.MinSize(1))

        broadcaster_table = reverb.Table(
            name=broadcaster_table_name,
            sampler=reverb.selectors.Fifo(),
            remover=reverb.selectors.Fifo(),
            max_size=1,
            rate_limiter=reverb.rate_limiters.MinSize(1))

        shutdown_table = reverb.Table(name=shutdown_table_name,
                                      sampler=reverb.selectors.Lifo(),
                                      remover=reverb.selectors.Fifo(),
                                      max_size=1,
                                      rate_limiter=reverb.rate_limiters.MinSize(1))
        self._server = reverb.Server([replay_table, model_table, shutdown_table, broadcaster_table], port=port)

        # The adder is used to insert observations into replay.
        address = f'localhost:{self._server.port}'
        self.client = reverb.Client(address)

        with tf.device(device_placement):
            dataset = datasets.make_reverb_dataset(
                table=replay_table_name,
                server_address=address,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
            ).as_numpy_iterator()
            if epsilon is None:
                epsilon = tf.Variable(0.05, trainable=False)

            # Create a target network.
            target_network = copy.deepcopy(policy_network)

            # Ensure that we create the variables before proceeding (maybe not needed).
            create_variables(policy_network, observation_spec)
            create_variables(target_network, observation_spec)

            checkpoint = True
            # The learner updates the parameters (and initializes them).
            self.learner = learning.DQNLearner(
                network=policy_network,
                target_network=target_network,
                discount=discount,
                importance_sampling_exponent=importance_sampling_exponent,
                learning_rate=learning_rate,
                target_update_period=target_update_period,
                dataset=dataset,
                logger=logger,
                checkpoint=checkpoint
            )
            if checkpoint:
                self._checkpointer = tf2_savers.Checkpointer(
                    directory=checkpoint_subpath,
                    objects_to_save=self.learner.state,
                    subdirectory='dqn_learner',)
            else:
                self._checkpointer = None
