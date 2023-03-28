import traceback
from acme import core
from acme.adders import reverb as adders
from acme.agents.tf import actors
import tree
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from grid2op import make
from grid2op.Action import TopologyChangeAndDispatchAction
from grid2op.Reward import GameplayReward, L2RPNReward, CombinedScaledReward
from dqn_args import parser
from utils import  get_actor_sigma, DQNnetwork, create_variables,convert_obs
import numpy as np
import pickle
import pytorch_actors
import reverb
import sonnet as snt
import sys
import tensorflow as tf
import time
import torch
import trfl
import zlib
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

    # 创建一个电网环境
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
    DQN_network = DQNnetwork(environment_grid.action_space, environment_grid.observation_space)
    # 根据电网环境创建网络
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

        # 配置adder
        adder = adders.NStepTransitionAdder(
            priority_fns={args["replay_table_name"]: lambda x: 1.},
            client=client,
            n_step=args["n_step"],
            discount=args["discount"])

        variable_source = ExternalVariableSource(client, args["model_table_name"], actor_id, args)
        variable_client = tf2_variable_utils.VariableClient(
            variable_source, {'policy': policy_network.variables}, update_period=args["actor_update_period"])

        # 创建FeedActor
        actor = actors.FeedForwardActor(behavior_network, adder=adder, variable_client=variable_client)

        # 创建pytorch actor
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
        # 将experience写入table中
        actor._adder._writer.append(dict(observation=observation,
                             start_of_episode=True),
                        partial_step=True)
        actor._adder._writer._add_first_called = True
        t_start_local = time.time()
        local_steps = 0

        while not done:
            local_steps += 1
            tstart = time.time()

            # 根据actor模型选取action
            if n_total_steps<args["pre_step"]:
                action=np.random.randint(0,action_spec)
            else:
                with tf.device(actor_device_placement):
                    action = actor.select_action(observation)
            action_con=DQN_network.convert_act(action)
            new_obs, reward, done, info = environment_grid.step(action_con)
            new_obs_con=convert_obs(new_obs)
            # print(new_obs_con)
            # 将experience写入table中
            if actor._adder._writer.episode_steps >= actor._adder.n_step:
                actor._adder._first_idx += 1
            actor._adder._last_idx += 1
            current_step = dict(
                # Observation was passed at the previous add call.
                action=np.int32(action),
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
            if done:
                dummy_step = tree.map_structure(np.zeros_like, current_step)
                actor._adder._writer.append(dummy_step)
                actor._adder._write_last()
                actor._adder.reset()

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
