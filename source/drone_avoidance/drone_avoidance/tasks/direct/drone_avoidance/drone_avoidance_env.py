# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import torch


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import SPHERE_MARKER_CFG, CUBOID_MARKER_CFG
from isaaclab.sensors import TiledCamera
import gymnasium as gym
import torch.nn.functional as F
from isaaclab.utils.math import subtract_frame_transforms
from dataclasses import replace

from .drone_avoidance_env_cfg import DroneAvoidanceEnvCfg


class DroneAvoidanceEnv(DirectRLEnv):
    cfg: DroneAvoidanceEnvCfg

    def __init__(self, cfg: DroneAvoidanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._obstacle_pos_w = torch.zeros(self.num_envs, self.cfg.num_obstacles, 3, device=self.device)
        self._collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reached_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_hold_time = torch.zeros(self.num_envs, device=self.device)
        self._goal_inside_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_reached_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_hold_update_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._obstacle_radius = torch.full(
            (self.num_envs, self.cfg.num_obstacles, 1),
            self.cfg.obstacle_radius,
            device=self.device,
        )
        self._prev_distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "near_obstacle",
                "collision",
                "alive",
                "upright",
                "action",
                "death",
                "progress_to_goal",
                "yaw_to_goal",
                "reached_goal",
                "lateral_vel",
                "forward_vel",
                "time",
                "speed_limit"
            ]
        }

        body_ids, _ = self.robot.find_bodies("body")
        self._body_id = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self.set_debug_vis(self.cfg.debug_vis)
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.obstacle = []
        for i in range(self.cfg.num_obstacles):
            obstacle_cfg = replace(
                self.cfg.obstacle_cfg,
                prim_path=f"/World/envs/env_.*/Obstacle_{i}",
            )
            obstacle = RigidObject(obstacle_cfg)
            self.obstacle.append(obstacle)
        self.depth_camera = TiledCamera(self.cfg.depth_camera)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.sensors["depth_camera"] = self.depth_camera
        for i, obstacle in enumerate(self.obstacle):
            self.scene.rigid_objects[f"obstacle_{i}"] = obstacle
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)
        throttle = self._actions[:, 0]
        thrust_ratio = torch.where(
            throttle >= 0.0,
            1.0 + throttle * (self.cfg.thrust_to_weight -1.0),
            1.0 + throttle,
        )
        self._thrust[:, 0, 2] = self._robot_weight * thrust_ratio
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
        

        depth = self.depth_camera.data.output["depth"]
        depth = torch.nan_to_num(depth, nan=self.cfg.depth_max_distance, posinf=self.cfg.depth_max_distance, neginf=self.cfg.depth_min_distance)
        depth = torch.clamp(depth, self.cfg.depth_min_distance, self.cfg.depth_max_distance)
        depth = (depth - self.cfg.depth_min_distance) / (
            self.cfg.depth_max_distance - self.cfg.depth_min_distance
        )
        
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        elif depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)

        depth_small = F.interpolate(
            depth,
            size=(self.cfg.depth_obs_height, self.cfg.depth_obs_width),
            mode="area",
        )
        depth_obs = depth_small.reshape(self.num_envs, -1)

        
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                desired_pos_b,
                depth_obs,

            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    
    def _update_goal_reached(self) -> None:
        needs_update = self._goal_hold_update_step != self.common_step_counter
        if not torch.any(needs_update):
            return
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.robot.data.root_pos_w, dim=1)

        inside_enter = distance_to_goal < self.cfg.reached_enter_radius
        outside_exit = distance_to_goal > self.cfg.reached_exit_radius

        goal_inside = self._goal_inside_buf.clone()
        goal_inside[inside_enter] = True
        goal_inside[outside_exit] = False

        goal_hold_time = torch.where(
            goal_inside,
            self._goal_hold_time + self.step_dt,
            torch.zeros_like(self._goal_hold_time),
        )

        goal_reached = goal_hold_time >= self.cfg.reached_hold_time

        self._goal_inside_buf[needs_update] = goal_inside[needs_update]
        self._goal_hold_time[needs_update] = goal_hold_time[needs_update]
        self._goal_reached_buf[needs_update] = goal_reached[needs_update]
        self._goal_hold_update_step[needs_update] = self.common_step_counter

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self.robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self.robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 4.0)
        distance_to_obstacle = torch.linalg.norm(self._obstacle_pos_w - self.robot.data.root_pos_w.unsqueeze(1), dim=2)
        min_distance_to_obstacle = torch.min(distance_to_obstacle, dim=1).values
        collision = torch.any(distance_to_obstacle < self.cfg.obstacle_radius + self.cfg.drone_radius, dim=1)
        near_obstacle = torch.clamp(
            self.cfg.near_obstacle_distance - min_distance_to_obstacle,
            min=0.0,
        ) / self.cfg.near_obstacle_distance
        alive = torch.ones(self.num_envs, device=self.device)
        upright = torch.clamp(-self.robot.data.projected_gravity_b[:, 2], min=0.0)
        action_penalty = torch.sum(torch.square(self._actions), dim=1)
        progress_to_goal = self._prev_distance_to_goal - distance_to_goal
        died = torch.logical_or(
            self.robot.data.root_pos_w[:, 2] < self.cfg.flight_z_min,
            self.robot.data.root_pos_w[:, 2] > self.cfg.flight_z_max,
        )

        desired_pos_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self._desired_pos_w,
        )
        yaw_to_goal = torch.atan2(desired_pos_b[:, 1], desired_pos_b[:, 0])
        heading_reward = torch.cos(yaw_to_goal)

        lateral_vel = torch.square(self.robot.data.root_lin_vel_b[:, 1])

        self._update_goal_reached()
        reached = self._goal_reached_buf

        goal_vec_w = self._desired_pos_w - self.robot.data.root_pos_w
        goal_dir_w = goal_vec_w / torch.linalg.norm(goal_vec_w, dim=1, keepdim=True).clamp_min(1e-6)

        forward_vel_to_goal = torch.sum(self.robot.data.root_lin_vel_w * goal_dir_w, dim=1)
        forward_vel_to_goal = torch.clamp(forward_vel_to_goal, min=0.0)

        speed = torch.linalg.norm(self.robot.data.root_lin_vel_b, dim=1)
        speed_over_limit = torch.clamp(speed - self.cfg.max_speed, min=0.0)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "near_obstacle": near_obstacle * self.cfg.near_obstacle_reward_scale * self.step_dt,
            "collision": collision.float() * self.cfg.collision_penalty,
            "alive": alive * self.cfg.alive_reward_scale * self.step_dt,
            "upright": upright * self.cfg.upright_reward_scale * self.step_dt,
            "action": action_penalty * self.cfg.action_penalty_scale * self.step_dt,
            "death": died.float() * self.cfg.death_penalty,
            "progress_to_goal": progress_to_goal * self.cfg.progress_reward_scale,
            "yaw_to_goal": heading_reward * self.cfg.yaw_reward_scale * self.step_dt,
            "reached_goal": reached.float() * self.cfg.reached_goal,
            "lateral_vel": lateral_vel * self.cfg.lateral_vel_reward_scale * self.step_dt,
            "forward_vel": forward_vel_to_goal * self.cfg.forward_vel_reward_scale * self.step_dt,
            "time": torch.ones(self.num_envs, device=self.device) * self.cfg.time_penalty_scale * self.step_dt,
            "speed_limit": torch.square(speed_over_limit) * self.cfg.speed_limit_reward_scale * self.step_dt

        }

        self._prev_distance_to_goal = distance_to_goal.detach()

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self.robot.data.root_pos_w[:, 2] < self.cfg.flight_z_min,
            self.robot.data.root_pos_w[:, 2] > self.cfg.flight_z_max,
        )

        distance_to_obstacle = torch.linalg.norm(self._obstacle_pos_w - self.robot.data.root_pos_w.unsqueeze(1), dim=2)
        collision = torch.any(distance_to_obstacle < self.cfg.obstacle_radius + self.cfg.drone_radius, dim=1)

        
        self._update_goal_reached()
        reached = self._goal_reached_buf

        self._collision_buf = collision
        self._died_buf = died
        self._reached_buf = reached

        terminated = torch.logical_or(died, collision)
        terminated2 = torch.logical_or(terminated, reached)
        return terminated2, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None :
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        final_distance_to_goal_each = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self.robot.data.root_pos_w[env_ids],
            dim=1,
        )
        final_distance_to_goal = final_distance_to_goal_each.mean()
        success_rate = torch.mean((self._goal_hold_time[env_ids] >= self.cfg.reached_hold_time).float())

        

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()

        num_reset_envs = len(env_ids)

        collision_count = torch.count_nonzero(self._collision_buf[env_ids]).item()
        died_count = torch.count_nonzero(self._died_buf[env_ids]).item()
        reached_count = torch.count_nonzero(self._reached_buf[env_ids]).item()
        timeout_count = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/success_rate"] = success_rate.item()
        extras["Metrics/collision_rate"] = collision_count / num_reset_envs
        extras["Metrics/died_rate"] = died_count / num_reset_envs
        extras["Metrics/timeout_rate"] = timeout_count / num_reset_envs

        extras["Episode_Termination/collision"] = collision_count
        extras["Episode_Termination/died"] = died_count
        extras["Episode Termination/reached"] = reached_count
        extras["Episode_Termination/time_out"] = timeout_count
        

        self.extras["log"].update(extras)

        self.robot.reset(env_ids) #type: ignore[arg-type]
        super()._reset_idx(env_ids) #type: ignore[arg-type]

        # if len(env_ids) == self.num_envs:
        #     self.episode_length_buf = torch.randint_like(
        #         self.episode_length_buf,
        #         high=int(self.max_episode_length),
        #     )

        

        self._actions[env_ids] = 0.0
        self._goal_hold_time[env_ids] = 0.0
        self._goal_inside_buf[env_ids] = False
        self._goal_reached_buf[env_ids] = False
        self._goal_hold_update_step[env_ids] = -1

        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(
            -self.cfg.goal_xy_range,
            self.cfg.goal_xy_range,
        )
        self._desired_pos_w[env_ids, :2] += self.terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(
            self.cfg.goal_z_min,
            self.cfg.goal_z_max,
        )
        start_pos_w = self.terrain.env_origins[env_ids].clone()
        start_pos_w[:, 2] = 1.0

        num_reset_envs = len(env_ids)
        num_candidates = self.cfg.obstacle_sample_attempts

        start_xy = start_pos_w[:, :2]
        goal_xy = self._desired_pos_w[env_ids, :2]
        path_xy = goal_xy - start_xy

        path_len = torch.linalg.norm(path_xy, dim=1, keepdim=True).clamp_min(1e-6)
        forward_dir = path_xy / path_len
        side_dir = torch.stack([-forward_dir[:, 1], forward_dir[:, 0]], dim=1)

        alpha = torch.empty(num_reset_envs, num_candidates, 1, device=self.device).uniform_(
            self.cfg.obstacle_path_alpha_min,
            self.cfg.obstacle_path_alpha_max,
        )

        lateral = torch.empty(num_reset_envs, num_candidates, 1, device=self.device).uniform_(
            -self.cfg.obstacle_lateral_range,
            self.cfg.obstacle_lateral_range,
        )

        z = torch.empty(num_reset_envs, num_candidates, 1, device=self.device).uniform_(
            self.cfg.obstacle_z_min,
            self.cfg.obstacle_z_max,
        )

        candidate_xy = (
            start_xy.unsqueeze(1)
            + alpha * path_xy.unsqueeze(1)
            + lateral * side_dir.unsqueeze(1)
        )

        candidates = torch.cat([candidate_xy, z], dim=-1)

        distance_to_start = torch.linalg.norm(candidate_xy - start_xy.unsqueeze(1), dim=-1)
        distance_to_goal = torch.linalg.norm(candidate_xy - goal_xy.unsqueeze(1), dim=-1)
        valid_candidates = torch.logical_and(
            distance_to_start > self.cfg.obstacle_start_clearance,
            distance_to_goal > self.cfg.obstacle_goal_clearance,
        )

        selected = torch.empty(
            num_reset_envs,
            self.cfg.num_obstacles,
            3,
            device=self.device,
        )

        for env_i in range(num_reset_envs):
            count = 0

            for cand_i in range(num_candidates):
                if not valid_candidates[env_i, cand_i].item():
                    continue

                candidate = candidates[env_i, cand_i]

                if count == 0:
                    selected[env_i, count] = candidate
                    count += 1
                else:
                    distances = torch.linalg.norm(selected[env_i, :count] - candidate, dim=1)
                    if torch.all(distances > self.cfg.obstacle_min_spacing).item():
                        selected[env_i, count] = candidate
                        count += 1

                if count == self.cfg.num_obstacles:
                    break

            if count < self.cfg.num_obstacles:
                selected[env_i, count:] = start_pos_w[env_i]
                selected[env_i, count:, 2] = self.cfg.flight_z_max + 5.0
        self._obstacle_pos_w[env_ids] = selected
        
        for i, obstacle in enumerate(self.obstacle):
            obstacle_state = obstacle.data.default_root_state[env_ids].clone()
            obstacle_state[:, :3] = self._obstacle_pos_w[env_ids, i]
            obstacle_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
            obstacle_state[:, 7:] = 0.0
            obstacle.write_root_pose_to_sim(obstacle_state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(obstacle_state[:, 7:], env_ids)

        
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids) #type: ignore[arg-type]
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids) #type: ignore[arg-type]

        if self.robot.num_joints > 0:
            joint_pos = self.robot.data.default_joint_pos[env_ids]
            joint_vel = self.robot.data.default_joint_vel[env_ids]
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids) #type: ignore[arg-type]

        self._prev_distance_to_goal[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self.robot.data.root_pos_w[env_ids],
            dim=1,
        )
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                goal_marker_cfg = CUBOID_MARKER_CFG.copy() # type: ignore[attr-defined]
                goal_marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                goal_marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(goal_marker_cfg)

            # if not hasattr(self, "obstacle_visualizer"):
            #     obstacle_marker_cfg = SPHERE_MARKER_CFG.copy()
            #     obstacle_marker_cfg.markers["sphere"].radius = self.cfg.obstacle_radius
            #     obstacle_marker_cfg.prim_path = "/Visuals/Command/obstacle"
            #     self.obstacle_visualizer = VisualizationMarkers(obstacle_marker_cfg)

            self.goal_pos_visualizer.set_visibility(True)
            # self.obstacle_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            # if hasattr(self, "obstacle_visualizer"):
            #     self.obstacle_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        # self.obstacle_visualizer.visualize(self._obstacle_pos_w.reshape(-1, 3))
        
