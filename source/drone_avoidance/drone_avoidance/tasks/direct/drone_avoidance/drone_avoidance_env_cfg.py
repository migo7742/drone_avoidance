# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils


from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg


@configclass
class DroneAvoidanceEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 4
    depth_obs_height = 64
    depth_obs_width = 64
    observation_space = 12 + depth_obs_height * depth_obs_width
    state_space = 0
    debug_vis = True
    

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),

    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=24.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    

    thrust_to_weight = 1.5
    moment_scale = 0.003

    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    obstacle_radius = 0.5
    num_obstacles = 5
    collision_penalty = -10.0
    near_obstacle_reward_scale = -2.0
    near_obstacle_distance = 1.5
    alive_reward_scale = 0.4
    upright_reward_scale = 1.0
    action_penalty_scale = -0.02
    death_penalty = -8.0
    progress_reward_scale = 8.0

    side_offset_min = -2.0
    side_offset_max = 2.0


    goal_xy_range = 8.0
    goal_z_min = 0.7
    goal_z_max = 1.8
    flight_z_min = 0.15
    flight_z_max = 2.5
    goal_radius = 0.3

    obstacle_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle",
        spawn=sim_utils.SphereCfg(
            radius=obstacle_radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.0))
    )

    depth_width = 64
    depth_height = 64
    depth_max_distance = 10.0

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_camera",
        update_period=0.02,
        height=depth_height,
        width=depth_width,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, depth_max_distance),

        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.02),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )