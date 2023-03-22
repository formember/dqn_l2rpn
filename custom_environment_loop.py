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

"""A simple agent-environment training loop."""
import tree
import itertools
import time
from typing import Optional

from acme import core
# Internal imports.
from acme.utils import counting
from acme.utils import loggers
import numpy as np

import dm_env

from utils import convert_obs


class CustomEnvironmentLoop(core.Worker):
    """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

    def __init__(
            self,
            environment,
            actor: core.Actor,
            DQN_network,
            counter: counting.Counter = None,
            logger: loggers.Logger = None,
            label: str = 'environment_loop',
    ):
        # Internalize agent and environment.
        self._DQN_network = DQN_network
        self._environment = environment
        self._actor = actor
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)
        self.begin_time = time.time()

    def run(self, num_episodes: Optional[int] = None):
        """Perform the run loop.

    Run the environment loop for `num_episodes` episodes. Each episode is itself
    a loop which interacts first with the environment to get an observation and
    then give that observation to the agent in order to retrieve an action. Upon
    termination of an episode a new episode will be started. If the number of
    episodes is not given then this will interact with the environment
    infinitely.

    Args:
      num_episodes: number of episodes to run the loop for. If `None` (default),
        runs without limit.
    """

        iterator = range(num_episodes) if num_episodes else itertools.count()
        avg_return = []

        for _ in iterator:
            # Reset any counts and start the environment.
            start_time = time.time()
            episode_steps = 0
            episode_return = 0
            done = False
            timestep = self._environment.reset()
            observation = convert_obs(timestep)
            # Make the first observation.
            self._actor._adder._writer.append(dict(observation=observation,
                                                   start_of_episode=True),
                                              partial_step=True)
            self._actor._adder._writer._add_first_called = True

            # Run an episode.
            while not done:
                # Generate an action from the agent's policy and step the environment.
                action = self._actor.select_action(observation)
                action_con = self._DQN_network.convert_act(action)
                new_obs, reward, done, info = self._environment.step(action_con)
                new_obs_con = convert_obs(new_obs)

                # Have the agent observe the timestep and let the actor update itself.
                if self._actor._adder._writer.episode_steps >= self._actor._adder.n_step:
                    self._actor._adder._first_idx += 1
                self._actor._adder._last_idx += 1
                current_step = dict(
                    # Observation was passed at the previous add call.
                    action=action,
                    reward=reward,
                    discount=np.float32(1.0),
                    **{}
                )
                self._actor._adder._writer.append(current_step)
                # Have the agent observe the timestep and let the actor update itself.

                self._actor._adder._writer.append(
                    dict(
                        observation=new_obs_con,
                        start_of_episode=False),
                    partial_step=True)
                self._actor._adder._write()
                if done:
                    # Complete the row by appending zeros to remaining open fields.
                    # TODO(b/183945808): remove this when fields are no longer expected to be
                    # of equal length on the learner side.
                    dummy_step = tree.map_structure(np.zeros_like, current_step)
                    self._actor._adder._writer.append(dummy_step)
                    self._actor._adder._write_last()
                    self._actor._adder.reset()
                self._actor.update()

                # Book-keeping.
                episode_steps += 1
                episode_return += reward
            avg_return.append(episode_return)

            # Record counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = episode_steps / (time.time() - start_time)
        mean_return = np.mean(avg_return) / episode_steps
        result = {
            'episode_length': episode_steps,
            'episode_return': np.mean(avg_return),
            'mean_return': mean_return,
            'steps_per_second': steps_per_second,
            'total_time_elapsed': time.time() - self.begin_time
        }
        result.update(counts)

        # Log the given results.
        self._logger.write(result)

# Internal class.
