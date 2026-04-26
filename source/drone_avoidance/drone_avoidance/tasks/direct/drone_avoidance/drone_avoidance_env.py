# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import torch


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import SPHERE_MARKER_CFG, CUBOID_MARKER_CFG

import gymnasium as gym

from isaaclab.utils.math import subtract_frame_transforms

from .drone_avoidance_env_cfg import DroneAvoidanceEnvCfg


class DroneAvoidanceEnv(DirectRLEnv):
    cfg: DroneAvoidanceEnvCfg

    def __init__(self, cfg: DroneAvoidanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._obstacle_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obstacle_radius = torch.full(
            (self.num_envs, 1),
            self.cfg.obstacle_radius,
            device=self.device,
        )
        

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "near_obstacle",
                "collision"
            ]
        }

        self._body_id = self.robot.find_bodies("body")[0]
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self.set_debug_vis(self.cfg.debug_vis)
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]


    def _apply_action(self) -> None:
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self._desired_pos_w,
        )
        obstacle_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self._obstacle_pos_w,
        )
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                desired_pos_b,
                obstacle_pos_b,
                self._obstacle_radius,

            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self.robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self.robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        distance_to_obstacle = torch.linalg.norm(self._obstacle_pos_w - self.robot.data.root_pos_w, dim=1)
        collision = distance_to_obstacle < self.cfg.obstacle_radius
        near_obstacle = torch.clamp(
            self.cfg.near_obstacle_distance - distance_to_obstacle,
            min=0.0,
        ) / self.cfg.near_obstacle_distance

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "near_obstacle": near_obstacle * self.cfg.near_obstacle_reward_scale * self.step_dt,
            "collision": collision.float() * self.cfg.collision_penalty,

        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self.robot.data.root_pos_w[:, 2] < 0.1,
            self.robot.data.root_pos_w[:, 2] > 2.0,
        )

        distance_to_obstacle = torch.linalg.norm(self._obstacle_pos_w - self.robot.data.root_pos_w, dim=1)
        collision = distance_to_obstacle < self.cfg.obstacle_radius
        
        self._collision_buf = collision

        terminated = torch.logical_or(died, collision)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self.robot.data.root_pos_w[env_ids],
            dim=1,
        ).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Episode_Termination/collision"] = torch.count_nonzero(self._collision_buf[env_ids]).item()
        self.extras["log"].update(extras)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf,
                high=int(self.max_episode_length),
            )

        self._actions[env_ids] = 0.0

        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self.terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        start_pos_w = self.terrain.env_origins[env_ids].clone()
        start_pos_w[:, 2] = 1.0

        alpha = torch.zeros(len(env_ids), 1, device=self.device).uniform_(0.35, 0.65)
        self._obstacle_pos_w[env_ids] = start_pos_w + alpha * (self._desired_pos_w[env_ids] - start_pos_w)

        side_offset = torch.zeros(len(env_ids), 1, device=self.device).uniform_(-0.4, 0.4)
        self._obstacle_pos_w[env_ids, 1:2] += side_offset
        self._obstacle_pos_w[env_ids, 2] = torch.zeros_like(self._obstacle_pos_w[env_ids, 2]).uniform_(0.7, 1.3)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                goal_marker_cfg = CUBOID_MARKER_CFG.copy()
                goal_marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                goal_marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)

            if not hasattr(self, "obstacle_visualizer"):
                obstacle_marker_cfg = SPHERE_MARKER_CFG.copy()
                obstacle_marker_cfg.markers["sphere"].radius = self.cfg.obstacle_radius
                obstacle_marker_cfg.prim_path = "/Visuals/Command/obstacle"
                self.obstacle_visualizer = VisualizationMarkers(obstacle_marker_cfg)

            self.goal_pos_visualizer.set_visibility(True)
            self.obstacle_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "obstacle_visualizer"):
                self.obstacle_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        self.obstacle_visualizer.visualize(self._obstacle_pos_w)
        


