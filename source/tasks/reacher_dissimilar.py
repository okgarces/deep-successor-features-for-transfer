# -*- coding: UTF-8 -*-
from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import torch
import numpy as np

from tasks.task import Task


class ReacherDissimilar(Task):
    def __init__(self, target_positions, task_index, torque_multipliers, filenames_mjfc='reacher_dissimilar_nf.cfg', include_target_in_state=False, device=None, seed=None, features_dim=None):
        super().__init__(device)

        self.target_positions = target_positions
        self.task_index = task_index
        self.target_pos = target_positions[task_index]
        self.include_target_in_state = include_target_in_state
        self.torque_multipliers = torque_multipliers
        self.filenames_mjfc = filenames_mjfc
        self.env = ReacherBulletEnv(self.target_pos, self.torque_multipliers[task_index], self.filenames_mjfc[task_index])
        self.seed = seed
        self.features_dim = features_dim

        print(f'Task_Index {self.task_index}, filename {self.filenames_mjfc[self.task_index]}, torque {self.torque_multipliers[self.task_index]}')
        
        # make the action lookup from integer to real action
        actions = [-1., 0., 1.]
        self.action_dict = dict()
        for a1 in actions:
            for a2 in actions:
                self.action_dict[len(self.action_dict)] = (a1, a2)
        
    def clone(self):
        return ReacherDissimilar(self.target_positions, self.task_index, self.include_target_in_state)
    
    def initialize(self):
        # if self.task_index == 0:
        #    self.env.render('human')
        state = self.env.reset(self.seed)
        if self.include_target_in_state:
            return np.concatenate([state.flatten(), self.target_pos])
        else:
            return state
    
    def action_count(self):
        return len(self.action_dict)
    
    def transition(self, action: int):
        action_int = int(action)
        real_action = self.action_dict[action_int]
        new_state, reward, done, _ = self.env.step(real_action)
        
        if self.include_target_in_state:
            return_state = np.concatenate([new_state, self.target_pos])
        else:
            return_state = new_state
            
        return return_state, reward, done
    
    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state: np.ndarray):
        return state.reshape((1, -1))
    
    def encode_dim(self):
        if self.include_target_in_state:
            return 6
        else:
            return 4
    
    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def features(self, state, action, next_state):
        phi = np.zeros((len(self.target_positions),))
        for index, target in enumerate(self.target_positions):
            delta = np.linalg.norm(self.env.robot.fingertip.pose().xyz()[:2] - target)
            phi[index] = 1. - 4. * delta
        return phi
    
    def feature_dim(self):
        if self.features_dim is not None:
            return self.features_dim
        return len(self.target_positions)
    
    def get_w(self):
        w = np.zeros((len(self.target_positions), 1))
        w[self.task_index, 0] = 1.0
        return w


class ReacherBulletEnv(BaseBulletEnv):

    def __init__(self, target, torque_multiplier, config_filename):
        self.robot = ReacherRobot(target, torque_multiplier, config_filename)
        BaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # sets self.to_target_vec
        
        delta = np.linalg.norm(self.robot.fingertip.pose().xyz() - np.array(self.robot.target.pose().xyz()))
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

    def __init__(self, target, torque_multiplier, config_filename):
        MJCFBasedRobot.__init__(self, config_filename, 'body0', action_dim=2, obs_dim=4)
        self.target_pos = target
        self.torque_multiplier = torque_multiplier

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
        assert (np.isfinite(a).all())
        # Initial Reacher has torque multiplier 0.05
        self.central_joint.set_motor_torque(self.torque_multiplier * float(np.clip(a[0], -1, +1)))
        self.elbow_joint.set_motor_torque(self.torque_multiplier * float(np.clip(a[1], -1, +1)))

    def calc_state(self):
        theta, self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        # target_x, _ = self.jdict["target_x"].current_position()
        # target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target.pose().xyz())
        return np.array([
            theta,
            self.theta_dot,
            self.gamma,
            self.gamma_dot
        ])
# 
#     def calc_potential(self):
#         return -100 * np.linalg.norm(self.to_target_vec)

