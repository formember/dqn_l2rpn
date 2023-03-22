import gc
from absl import app
from absl import flags
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from grid2op import make
from grid2op.Action import TopologyChangeAndDispatchAction
from grid2op.Reward import CombinedScaledReward, GameplayReward, L2RPNReward
from grid2op.gym_compat import GymEnv, BoxGymObsSpace, BoxGymActSpace
from l2rpn_baselines.PPO_SB3 import remove_non_usable_attr

from dm_control import suite
from d4pg_args import parser
from dqn_learner import DQN_learner
from multiprocessing import Process
from typing import Mapping, Sequence
from utils import make_environment, Q, input_size_from_obs_spec, Q_opt, Q_decompress, Q_compress, DQNnetwork
import acme
import argparse
import dm_env
import gym
import numpy as np
import reverb
import sonnet as snt
import sys
import tensorflow as tf
import threading
from concurrent import futures
import time
import trfl
from custom_environment_loop import CustomEnvironmentLoop
import zlib
import pickle
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import pytorch_actors
import torch
import zstd
import msgpack

# For debugging
#tf.debugging.set_log_device_placement(True)

def get_shutdown_status(client, shutdown_table_name):
    sample = next(client.sample(shutdown_table_name))[0]
    return int(sample.data[0])

def get_weights_to_broadcast(client, broadcast_table_name):
    print("Waiting for weights", time.time())
    sample = next(client.sample(broadcast_table_name))[0]
    print("Received weights", time.time())
    return sample.data
    
if __name__ == "__main__":
    args = parser.parse_args()

    print(vars(args))

    # Initialize environment
    environment_grid = make(args.taskstr,
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
    #                                                    attr_to_keep=obs_attr_to_keep
    #                                                    )
    # environment_gym.action_space.close()
    # default_act_attr_to_keep = ["redispatch", "curtail", "set_storage"]
    # act_attr_to_keep = remove_non_usable_attr(environment_grid, default_act_attr_to_keep)
    # environment_gym.action_space = BoxGymActSpace(environment_gym.init_env.action_space,
    #                                               attr_to_keep=act_attr_to_keep
    #                                               )
    # # myenv=MyEnv(environment,environment_grid)
    # environment = wrappers.GymWrapper(environment_gym)
    # environment = wrappers.SinglePrecisionWrapper(environment)

    # environment_spec = specs.make_environment_spec(environment)
    model_sizes = tuple([int(x) for x in args.model_str.split(",")])
    DQN_network = DQNnetwork(environment_grid.action_space, environment_grid.observation_space)
    policy_network = DQN_network.make_networks(placement=args.learner_device_placement,
                                               policy_layer_sizes=model_sizes)["policy"]
    observation_spec = DQN_network.observation_size
    action_spec = DQN_network.action_size

    # Create pytorch model (we send the state dict to the actors)
    pytorch_model = pytorch_actors.create_model(observation_spec,
                                                action_spec,
                                                policy_layer_sizes=model_sizes)
    
    address = "localhost:%d" % args.port
    client = reverb.Client(address)

    def quantize_and_broadcast_weights(weights, id):

        print("Broadcasting weights", time.time())
        
        # Quantize weights artificially
        tstart = time.time()
        weights = [Q(x, args.quantize_communication) for x in weights]
        #weights = [Q_opt(x, args.quantize_communication) for x in weights]
        print("Quantized (t=%f)" % (time.time()-tstart))

        # Load weights into pytorch model
        tstart = time.time()
        pytorch_actors.pytorch_model_load_state_dict(pytorch_model, weights)
        print("Load weights (t=%f)" % (time.time()-tstart))

        # Quantize
        tstart = time.time()
        quantized_actor = pytorch_actors.pytorch_quantize(pytorch_model, args.quantize)
        print("Pytorch quantized (t=%f)" % (time.time()-tstart))

        # State dict
        state_dict = quantized_actor.state_dict()
        if args.weight_compress != 0:
            for k,v in state_dict.items():
                state_dict[k] = Q_compress(state_dict[k], args.weight_compress)
        state_dict["id"] = id

        # Send over packed params to avoid overhead
        tstart = time.time()
        weights = [pickle.dumps(state_dict)]
        if args.compress:
            weights = [zlib.compress(x) for x in weights]
        weights = [np.fromstring(x, dtype=np.uint8) for x in weights]
        print("Compress %f" % (time.time()-tstart))


        client.insert(weights, 
                      {args.model_table_name : 1.0})

        print("Done broadcasting", time.time())

    c = 0
    while True:        
        should_shutdown = get_shutdown_status(client, args.shutdown_table_name)
        sys.stdout.flush()
        if should_shutdown:
            break

        weights = get_weights_to_broadcast(client, args.broadcaster_table_name)
        quantize_and_broadcast_weights(weights, c)
        
        c += 1
