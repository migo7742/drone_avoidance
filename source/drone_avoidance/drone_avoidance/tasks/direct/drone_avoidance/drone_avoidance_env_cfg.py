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
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg


@configclass
class DroneAvoidanceEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 4
    depth_obs_height = 72
    depth_obs_width = 128
    observation_space = 12 + depth_obs_height * depth_obs_width
    state_space = 0
    debug_vis = False
    

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
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/migo/drone_avoidance/drone.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=24.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    

    thrust_to_weight = 4.7
    moment_scale = 0.04
    drone_radius = 0.15

    lin_vel_reward_scale = -0.1
    ang_vel_reward_scale = -0.5

    goal_xy_range = 8.0
    goal_z_min = 1.0
    goal_z_max = 1.6
    goal_radius = 0.3
    reached_goal = 160.0
    reached_enter_radius = 0.3
    reached_exit_radius = 0.45
    reached_hold_time = 2.0
    distance_to_goal_reward_scale = 40.0

    alive_reward_scale = 1.0
    upright_reward_scale = 5.0
    action_penalty_scale = -0.02
    death_penalty = -100.0
    progress_reward_scale = 20.0
    lateral_vel_reward_scale = -2.0
    forward_vel_reward_scale = 2.0
    time_penalty_scale = -1.0
    yaw_reward_scale = 5.0
    max_speed = 2.0
    speed_limit_reward_scale = -3.0
    
    num_obstacles = 20
    collision_penalty = -180.0
    near_obstacle_reward_scale = -7.0
    near_obstacle_distance = 2.0
    obstacle_radius = 0.35
    obstacle_path_alpha_min = 0.15
    obstacle_path_alpha_max = 0.9
    obstacle_lateral_range = 3.0
    obstacle_z_min = 0.7
    obstacle_z_max = 2.5
    obstacle_min_spacing = 0.9
    obstacle_sample_attempts = 400
    obstacle_start_clearance = 0.8
    obstacle_goal_clearance = 0.8
    
    flight_z_min = 0.2
    flight_z_max = 5.6
    
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

    depth_width = 128
    depth_height = 72
    depth_max_distance = 10.0
    depth_min_distance = 0.28

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_camera_mount/front_camera",
        update_period=1.0 / 30.0,
        height=depth_height,
        width=depth_width,
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.04,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(depth_min_distance, depth_max_distance),

        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
