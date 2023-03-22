# import pytorch_actors
# from absl import app
# from absl import flags
import traceback
from acme import core
# from acme import datasets
# from acme import specs
from acme import specs
# from acme import types
# from acme import types
# from acme import wrappers
from acme.adders import reverb as adders
# from acme.agents import agent
# from acme.agents.tf import actors
from acme.agents.tf import actors
# from acme.agents.tf import d4pg
# from acme.agents.tf.d4pg import learning
# from acme.tf import networks
from acme.tf import networks
import tree
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from grid2op import make
from grid2op.Action import TopologyChangeAndDispatchAction
from grid2op.Reward import GameplayReward, L2RPNReward, CombinedScaledReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace
from l2rpn_baselines.PPO_SB3 import remove_non_usable_attr

# from acme.utils import counting
# from acme.utils import loggers
# from acme.utils import loggers
from d4pg_args import parser
# from dm_control import suite
# from multiprocessing import Process
# from typing import Mapping, Sequence
from utils import  get_actor_sigma, DQNnetwork, create_variables,convert_obs
# import acme
# import argparse
# import copy
# import dm_env
# import gc
# import gym
# import numpy as np
import numpy as np
import pickle
import pytorch_actors
import reverb
import reverb
import sonnet as snt
import sonnet as snt
import sys
import sys
import sys
import tensorflow as tf
import time
import time
import torch
import trfl
import zlib
import zstd
from acme import wrappers
torch.set_num_threads(1)
cpus = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpus)


class ExternalVariableSource(core.VariableSource):

    def __init__(self, reverb_table, model_table_name, actor_id, args):
        self.reverb_table = reverb_table
        self.model_table_name = model_table_name
        self.actor_id = actor_id
        self.prev_sample = None
        self.args = args
        self.cached = None

    def get_variables(self, names):
        # if self.cached is not None:
        #  return self.cached

        # Pull from reverb table
        tstart = time.time()
        sample = next(self.reverb_table.sample(self.model_table_name))[0]
        tend = time.time()

        # Decode sample
        d = [x.tobytes() for x in sample.data]

        try:
            if self.args["compress"]:
                d = [zlib.decompress(x) for x in d]
            tdecompress = time.time()
            decoded = [pickle.loads(x) for x in d]
            tdecode = time.time()
            print("Pull time: %f, Decompress/tobytes time: %f, Deserialize time: %f" % (
            tend - tstart, tdecompress - tend, tdecode - tdecompress))
            return decoded
        except:
            traceback.print_exc()
            pass
        return []


class IntraProcessTFToPyTorchVariableSource(core.VariableSource):
    def __init__(self, tf_model):
        self.tf_model = tf_model

    def get_variables(self, name):
        res = [tf2_utils.to_numpy(v) for v in self.tf_model.variables]
        return res


def get_shutdown_status(client, shutdown_table_name):
    sample = next(client.sample(shutdown_table_name))[0]
    return int(sample.data[0])


def get_rapaly_data(client):
    sample = next(client.sample("priority_table"))[0]
    return sample


def actor_main(actor_id, args):
    print("Starting actor %d" % actor_id)

    address = "localhost:%d" % args["port"]
    client = reverb.Client(address)
    actor_device_placement = args["actor_device_placement"]
    actor_device_placement = "%s:0" % (actor_device_placement)

    model_sizes = tuple([int(x) for x in args["model_str"].split(",")])

    # Create network / env
    environment_grid = make(args["taskstr"],
                            action_class=TopologyChangeAndDispatchAction,
                            reward_class=CombinedScaledReward)

    # Only load 128 steps in ram
    environment_grid.chronics_handler.set_chunk_size(128)

    # Register custom reward for training
    try:
        # change of name in grid2op >= 1.2.3
        cr = environment_grid._reward_helper.template_reward
    except AttributeError as nm_exc_:
        cr = environment_grid.reward_helper.template_reward
    # cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 1.0)
    # cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    cr.addReward("l2rpn", L2RPNReward(), 2.0 / float(environment_grid.n_line))
    # Initialize custom rewards
    cr.initialize(environment_grid)
    # Set reward range to something managable
    cr.set_range(-1.0, 1.0)
    # environment_gym = GymEnv(environment_grid)
    # glop_obs = environment_grid.reset()
    # environment_gym.observation_space.close()
    # obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
    #                     "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
    #                     "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
    #                     "storage_power", "storage_charge"]
    # environment_gym.observation_space = BoxGymObsSpace(environment_gym.init_env.observation_space,
    #                                                attr_to_keep=obs_attr_to_keep
    #                                                )
    # environment_gym.action_space.close()
    # default_act_attr_to_keep = ["redispatch", "curtail", "set_storage"]
    # act_attr_to_keep = remove_non_usable_attr(environment_grid, default_act_attr_to_keep)
    # environment_gym.action_space = BoxGymActSpace(environment_gym.init_env.action_space,
    #                                           attr_to_keep=act_attr_to_keep
    #                                           )
    # # myenv=MyEnv(environment,environment_grid)
    # environment = wrappers.GymWrapper(environment_gym)
    # environment = wrappers.SinglePrecisionWrapper(environment)
    # environment_spec = specs.make_environment_spec(environment)
    #
    # act_spec = environment_spec.actions
    # obs_spec = environment_spec.observations
    DQN_network = DQNnetwork(environment_grid.action_space, environment_grid.observation_space)
    policy_network = DQN_network.make_networks(placement=args["learner_device_placement"],
                                               policy_layer_sizes=model_sizes)["policy"]
    observation_spec=DQN_network.observation_size
    action_spec=DQN_network.action_size
    with tf.device(actor_device_placement):
        emb_spec = create_variables(tf.identity, observation_spec)
        tf2_utils.create_variables(policy_network, [emb_spec])
        epsilon = tf.Variable(get_actor_sigma(args["sigma"], args["actor_id"], args["n_actors"]), trainable=False)
        behavior_network = snt.Sequential([
            policy_network,
            lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
        ])

        # Set up actor
        adder = adders.NStepTransitionAdder(
            priority_fns={args["replay_table_name"]: lambda x: 1.},
            client=client,
            n_step=args["n_step"],
            discount=args["discount"])

        variable_source = ExternalVariableSource(client, args["model_table_name"], actor_id, args)
        variable_client = tf2_variable_utils.VariableClient(
            variable_source, {'policy': policy_network.variables}, update_period=args["actor_update_period"])

        # Create Feed actor
        actor = actors.FeedForwardActor(behavior_network, adder=adder, variable_client=variable_client)

        # Create pytorch actor
        pytorch_adder = adders.NStepTransitionAdder(
            priority_fns={args["replay_table_name"]: lambda x: 1.},
            client=client,
            n_step=args["n_step"],
            discount=args["discount"])

        pytorch_model = pytorch_actors.create_model(observation_spec,
                                                    action_spec,
                                                    policy_layer_sizes=model_sizes)
        pytorch_variable_source = ExternalVariableSource(client, args["model_table_name"], actor_id, args)
        pytorch_variable_client = pytorch_actors.TFToPyTorchVariableClient(
            pytorch_variable_source, pytorch_model, update_period=args["actor_update_period"])
        pytorch_actor = pytorch_actors.FeedForwardActor(pytorch_model,
                                                        adder=pytorch_adder,
                                                        variable_client=pytorch_variable_client,
                                                        q=args["quantize"],
                                                        args=args)

    actor = {
        "tensorflow": actor,
        "pytorch": pytorch_actor,
    }[args["inference_mode"]]

    # Main actor loop
    t_start = time.time()
    n_total_steps = 0
    while True:
        should_shutdown = get_shutdown_status(client, args["shutdown_table_name"])
        sys.stdout.flush()
        if should_shutdown:
            break
        timestep = environment_grid.reset()
        observation = convert_obs(timestep)
        episode_return = 0
        done=False
        actor._adder._writer.append(dict(observation=observation,
                             start_of_episode=True),
                        partial_step=True)
        actor._adder._writer._add_first_called = True
        t_start_local = time.time()
        local_steps = 0

        while not done:
            local_steps += 1
            tstart = time.time()

            # Generate an action from the agent's policy and step the environment.
            with tf.device(actor_device_placement):
                action = actor.select_action(observation)
            action_con=DQN_network.convert_act(action)
            new_obs, reward, done, info = environment_grid.step(action_con)
            new_obs_con=convert_obs(new_obs)
            print(reward)
            print(action)
            # print(new_obs_con)
            if actor._adder._writer.episode_steps >= actor._adder.n_step:
                actor._adder._first_idx += 1
            actor._adder._last_idx += 1
            current_step = dict(
                # Observation was passed at the previous add call.
                action=action.astype(np.int32),
                reward=reward,
                discount=np.float32(1.0),
                **{}
            )
            actor._adder._writer.append(current_step)
            # Have the agent observe the timestep and let the actor update itself.

            actor._adder._writer.append(
                dict(
                    observation=new_obs_con,
                    start_of_episode=False),
                partial_step=True)
            actor._adder._write()
            # print(list(client.sample('priority_table', num_samples=1)))
            # print(1)
            if done:
                # Complete the row by appending zeros to remaining open fields.
                # TODO(b/183945808): remove this when fields are no longer expected to be
                # of equal length on the learner side.
                dummy_step = tree.map_structure(np.zeros_like, current_step)
                actor._adder._writer.append(dummy_step)
                actor._adder._write_last()
                actor._adder.reset()
            # actor.observe(action, next_timestep=timestep)

            episode_return +=reward
            n_total_steps += 1

            tend = time.time()

            print("Step time: %f" % (tend - tstart))

            # Update the actor
            # print(args)
            if n_total_steps * args["n_actors"] >= args["min_replay_size"]:
                actor.update()

        steps_per_second = n_total_steps / (time.time() - t_start)
        local_steps_per_second = local_steps / (time.time() - t_start_local)
        print("Actor %d finished timestep (r=%f) (steps_per_second=%f) (local_steps_per_second=%f)" % (
        actor_id, float(episode_return), steps_per_second, local_steps_per_second))
    print("Actor %d shutting down" % (actor_id))


if __name__ == "__main__":
    args = parser.parse_args()
    print(vars(args))
    actor_main(args.actor_id, vars(args))
