# -*- coding: UTF-8 -*-
from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import torch
import numpy as np

from tasks.task import Task
from utils.torch import get_torch_device


class Reacher(Task):
    
    def __init__(self, target_positions, task_index, include_target_in_state=False):
        self.target_positions = target_positions
        self.task_index = task_index
        self.target_pos = target_positions[task_index]
        self.include_target_in_state = include_target_in_state
        self.env = ReacherBulletEnv(self.target_pos)
        
        # make the action lookup from integer to real action
        actions = [-1., 0., 1.]
        self.action_dict = dict()
        for a1 in actions:
            for a2 in actions:
                self.action_dict[len(self.action_dict)] = (a1, a2)

        self.device = get_torch_device()
        
    def clone(self):
        return Reacher(self.target_positions, self.task_index, self.include_target_in_state)
    
    def initialize(self):
        # if self.task_index == 0:
        #    self.env.render('human')
        state = torch.tensor(self.env.reset()).to(self.device)
        if self.include_target_in_state:
            return torch.concat([state.flatten(), self.target_pos]).to(self.device)
        else:
            return state
    
    def action_count(self):
        return len(self.action_dict)
    
    def transition(self, action: torch.Tensor):
        action_int = int(action)
        real_action = self.action_dict[action_int]
        new_state, reward, done, _ = self.env.step(real_action)

        new_state = torch.tensor(new_state).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        
        if self.include_target_in_state:
            return_state = torch.concat([new_state, self.target_pos])
        else:
            return_state = new_state
            
        return return_state, reward, done
    
    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        return torch.tensor(state).reshape((1, -1)).to(self.device)
    
    def encode_dim(self):
        if self.include_target_in_state:
            return 6
        else:
            return 4
    
    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def features(self, state, action, next_state):
        phi = torch.zeros((len(self.target_positions),)).to(self.device)
        for index, target in enumerate(self.target_positions):
            delta = torch.linalg.norm(torch.tensor(self.env.robot.fingertip.pose().xyz()[:2]).to(self.device) - torch.tensor(target).to(self.device))
            phi[index] = 1. - 4. * delta
        return phi
    
    def feature_dim(self):
        return len(self.target_positions)
    
    def get_w(self):
        w = torch.zeros((len(self.target_positions), 1)).to(self.device)
        w[self.task_index, 0] = 1.0
        return w


class ReacherBulletEnv(BaseBulletEnv):

    def __init__(self, target):
        self.robot = ReacherRobot(target)
        BaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec
        
        delta = torch.linalg.norm(
            torch.Tensor(self.robot.fingertip.pose().xyz()) - np.array(self.robot.target.pose().xyz()))
        reward = 1. - 4. * delta
        self.HUD(state, a, False)
        
        return state, reward, False, {}

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)


class ReacherRobot(MJCFBasedRobot):
    TARG_LIMIT = 0.27

    def __init__(self, target):
        MJCFBasedRobot.__init__(self, 'reacher.xml', 'body0', action_dim=2, obs_dim=4)
        self.target_pos = target

    def robot_specific_reset(self, bullet_client):
        self.jdict["target_x"].reset_current_position(self.target_pos[0], 0)
        self.jdict["target_y"].reset_current_position(self.target_pos[1], 0)
        self.fingertip = self.parts["fingertip"]
        self.target = self.parts["target"]
        self.central_joint = self.jdict["joint0"]
        self.elbow_joint = self.jdict["joint1"]
        self.central_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
        self.elbow_joint.reset_current_position(self.np_random.uniform(low=-3.14 / 2, high=3.14 / 2), 0)

    def apply_action(self, a):
        # Here np is ok. To apply a tuple real action to env.
        assert (np.isfinite(a).all())
        self.central_joint.set_motor_torque(0.05 * float(np.clip(a[0], -1, +1)))
        self.elbow_joint.set_motor_torque(0.05 * float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        theta, self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        # target_x, _ = self.jdict["target_x"].current_position()
        # target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return torch.Tensor([
            theta,
            self.theta_dot,
            self.gamma,
            self.gamma_dot
        ])
# 
#     def calc_potential(self):
#         return -100 * np.linalg.norm(self.to_target_vec)

