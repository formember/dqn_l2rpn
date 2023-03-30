import sys
import torch
import pprint
from collections import OrderedDict
from typing import Mapping, Sequence, Optional
import sys
from acme.wrappers import base
import tree

from absl import app
from absl import flags
import acme
import reverb
try:
    import tensorflow as tf
    _CAN_USE_TENSORFLOW = True
except ImportError:
    _CAN_USE_TENSORFLOW = False
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf.networks import duelling
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import dm_env
import gym
import numpy as np
import sonnet as snt
from dm_control import suite

from multiprocessing import Process
import threading
import argparse

from acme.wrappers import gym_wrapper
from acme.wrappers import atari_wrapper
from acme.tf.networks import base

import bsuite
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

Images = tf.Tensor
QValues = tf.Tensor
Logits = tf.Tensor
Value = tf.Tensor


def get_actor_sigma(sigma_max, actor_id, n_actors):
    sigmas = list(np.arange(0, sigma_max, (sigma_max - 0) / n_actors))
    print(sigmas)
    return sigmas[actor_id - 1]


def Q_compress(W, n):
    assert (n == 8)

    W = W.numpy()
    W_orig = W
    if n >= 32:
        return W
    assert (len(W.shape) <= 2)
    range = np.max(W)-np.min(W)
    d = range / (2 ** (n - 1))
    if d == 0:
        return W
    z = -np.min(W, 0) // d
    W = np.rint(W / d)

    W_q = torch.from_numpy(W).char()
    return d, W_q


def Q_decompress(V, n):
    return V[0] * V[1].float()


def Q(W, n):
    if n >= 32:
        return W
    assert (len(W.shape) <= 2)
    range = np.abs(np.min(W)) + np.abs(np.max(W))
    d = range / (2 ** (n))
    if d == 0:
        return W
    z = -np.min(W, 0) // d
    W = np.rint(W / d)
    W = d * (W)
    return W


def Q_opt(W, n, intervals=100):
    return Q(W, n)
    if n == 32:
        return Q(W, n)
    best = Q(W, n)
    best_err = np.mean(np.abs((best - W)).flatten())
    first_err = best_err
    minW, maxW = np.min(W), np.max(W)
    max_abs = max(abs(minW), abs(maxW))
    for lim in np.arange(0, max_abs, max_abs / intervals):
        W_clipped = np.clip(W, -lim, lim)
        W_clipped_Q = Q(W_clipped, n)
        mse = np.mean(np.abs((W_clipped_Q - W)).flatten())
        if mse < best_err:
            # print("New best err: (%f->%f) at clip %f (W_min=%f, W_max=%f)" % (best_err, mse, lim, minW, maxW))
            best_err = mse
            best = W_clipped_Q
    print("Opted: %f->%f err" % (first_err, best_err))
    return best

def input_size_from_obs_spec(env_spec):
    if hasattr(env_spec, "shape"):
        return int(np.prod(env_spec.shape))
    if type(env_spec) == OrderedDict:
        return int(sum([input_size_from_obs_spec(x) for x in env_spec.values()]))
    try:
        return int(sum([input_size_from_obs_spec(x) for x in env_spec]))
    except:
        assert (0)


def input_from_obs(observation):
    observation = tf2_utils.add_batch_dim(observation)
    observation = tf2_utils.batch_concat(observation)
    return tf2_utils.to_numpy(observation)


# The default settings in this network factory will work well for the
# MountainCarContinuous-v0 task but may need to be tuned for others. In
# particular, the vmin/vmax and num_atoms hyperparameters should be set to
# give the distributional critic a good dynamic range over possible discounted
# returns. Note that this is very different than the scale of immediate rewards.


class DQNnetwork(AgentWithConverter):
    def __init__(self, action_space, observation_space):
        # Call parent constructor
        if not _CAN_USE_TENSORFLOW:
            raise RuntimeError("Cannot import tensorflow, this function cannot be used.")
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)
        self.action_space.filter_action(self._filter_action)
        self.action_size = self.action_space.size()
        self.obs_space = observation_space
        self.observation_size = self.obs_space.size_obs()

    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= MAX_ELEM:
            return True
        return False

    def convert_act(self, action):
        return super().convert_act(action)

    def make_networks(
            self,
            policy_layer_sizes: Sequence[int] = (2048, 2048, 2048),
            critic_layer_sizes: Sequence[int] = (512, 512, 256),
            vmin: float = -150.,
            vmax: float = 150.,
            num_atoms: int = 51,
            placement: str = "CPU",
    ) -> Mapping[str, types.TensorTransformation]:
        """Creates the networks used by the agent."""

        with tf.device(placement):
            # Get total number of action dimensions from action spec.
            num_dimensions = self.action_size

            # Create the shared observation network; here simply a state-less operation.
            # policy_layer_sizes=(self.observation_size*2,self.observation_size,896,512)
            # Create the policy network.
            uniform_initializer = tf.initializers.VarianceScaling(
                distribution='uniform', mode='fan_out', scale=0.333)
            network = snt.Sequential([
                snt.nets.MLP(
                    policy_layer_sizes,
                    w_init=uniform_initializer,
                    activation=tf.nn.relu,
                    activate_final=True
                )])
            policy_network = snt.Sequential([
                network,
                # networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
                networks.NearZeroInitializedLinear(num_dimensions),
                # tf.nn.max_pool_with_argmax
            ])


            return {
                'policy': policy_network,
            }

    def my_act(self, transformed_observation, reward, done=False):
        pass



def create_variables(
    network: snt.Module,
    input_spec
) -> Optional[tf.TensorSpec]:
  """Builds the network with dummy inputs to create the necessary variables.

  Args:
    network: Sonnet Module whose variables are to be created.
    input_spec: list of input specs to the network. The length of this list
      should match the number of arguments expected by `network`.

  Returns:
    output_spec: only returns an output spec if the output is a tf.Tensor, else
        it doesn't return anything (None); e.g. if the output is a
        tfp.distributions.Distribution.
  """
  from acme.tf.utils import squeeze_batch_dim,add_batch_dim,zeros_like
  # Create a dummy observation with no batch dimension.
  dummy_input = zeros_like([tf.convert_to_tensor([0.]*input_spec)])

  # If we have an RNNCore the hidden state will be an additional input.
  if isinstance(network, snt.RNNCore):
    initial_state = squeeze_batch_dim(network.initial_state(1))
    dummy_input += [initial_state]

  # Forward pass of the network which will create variables as a side effect.
  dummy_output = network(*add_batch_dim(dummy_input))

  # Evaluate the input signature by converting the dummy input into a
  # TensorSpec. We then save the signature as a property of the network. This is
  # done so that we can later use it when creating snapshots. We do this here
  # because the snapshot code may not have access to the precise form of the
  # inputs.
  input_signature = tree.map_structure(
      lambda t: tf.TensorSpec((None,) + t.shape, t.dtype), dummy_input)
  network._input_signature = input_signature  # pylint: disable=protected-access

  def spec(output):
    # If the output is not a Tensor, return None as spec is ill-defined.
    if not isinstance(output, tf.Tensor):
      return None
    # If this is not a scalar Tensor, make sure to squeeze out the batch dim.
    if tf.rank(output) > 0:
      output = squeeze_batch_dim(output)
    return tf.TensorSpec(output.shape, output.dtype)

  return tree.map_structure(spec, dummy_output)


def convert_obs(observation):
    li_vect = []
    for el in observation.attr_list_vect:
        v = observation._get_array_from_attr_name(el).astype(np.float32)
        v_fix = np.nan_to_num(v)
        v_norm = np.linalg.norm(v_fix)
        if v_norm > 1e6:
            v_res = (v_fix / v_norm) * 10.0
        else:
            v_res = v_fix
        li_vect.append(v_res)
    return np.concatenate(li_vect)