import gym
from grid2op.gym_compat import GymEnv
class MyEnv(gym.Env):
    def __init__(self, gymenv,gridenv):
        self.gridenv = gridenv
        import grid2op
        from grid2op.gym_compat import GymEnv
        from grid2op.gym_compat import ScalerAttrConverter, ContinuousToDiscreteConverter, MultiToTupleConverter

        # 2. create the gym environment
        self.env_gym = gymenv
        obs_gym = self.env_gym.reset()

        # 3. (optional) customize it (see section above for more information)
        ## customize action space
        self.env_gym.action_space = self.env_gym.action_space.ignore_attr("set_bus").ignore_attr("set_line_status")
        self.env_gym.action_space = self.env_gym.action_space.reencode_space("redispatch",
                                                                             ContinuousToDiscreteConverter(nb_bins=11)
                                                                             )
        self.env_gym.action_space = self.env_gym.action_space.reencode_space("change_bus", MultiToTupleConverter())
        self.env_gym.action_space = self.env_gym.action_space.reencode_space("change_line_status",
                                                                             MultiToTupleConverter())
        self.env_gym.action_space = self.env_gym.action_space.reencode_space("redispatch", MultiToTupleConverter())
        a = {k: v for k, v in self.env_gym.action_space.items()}
        self.action_space = gym.spaces.Dict(a)
        ## customize observation space
        ob_space = self.env_gym.observation_space
        ob_space = ob_space.keep_only_attr(["rho", "gen_p", "load_p", "topo_vect", "actual_dispatch"])
        ob_space = ob_space.reencode_space("actual_dispatch",
                                           ScalerAttrConverter(substract=0.,
                                                               divide=self.gridenv.gen_pmax
                                                               )
                                           )
        ob_space = ob_space.reencode_space("gen_p",
                                           ScalerAttrConverter(substract=0.,
                                                               divide=self.gridenv.gen_pmax
                                                               )
                                           )
        ob_space = ob_space.reencode_space("load_p",
                                           ScalerAttrConverter(substract=obs_gym["load_p"],
                                                               divide=0.5 * obs_gym["load_p"]
                                                               )
                                           )
        self.env_gym.observation_space = ob_space

        # 4. specific to rllib
        self.action_space = self.env_gym.action_space
        self.observation_space = self.env_gym.observation_space

        # 4. bis: to avoid other type of issues, we recommend to build the action space and observation
        # space directly from the spaces class.
        d = {k: v for k, v in self.env_gym.observation_space.spaces.items()}
        self.observation_space = gym.spaces.Dict(d)
        a = {k: v for k, v in self.env_gym.action_space.items()}
        self.action_space = gym.spaces.Dict(a)

    def reset(self):
        obs = self.env_gym.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env_gym.step(action)
        return obs, reward, done, info