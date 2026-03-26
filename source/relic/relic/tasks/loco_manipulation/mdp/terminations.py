# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Functions specific to the interlimb loco-manipulation environments."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

def outlier_detected(env: ManagerBasedRLEnv, threshold: float = 1000.0) -> torch.Tensor:
    """Terminates the environment if actions or base velocities explode."""
    
    # 1. Check for Exploding Actions
    actions = env.action_manager.action
    action_exploded = torch.any(torch.abs(actions) > threshold, dim=-1)
    action_nan = torch.any(torch.isnan(actions) | torch.isinf(actions), dim=-1)
    
    # 2. Check for Exploding Robot Base Velocities (A common symptom of physics explosions)
    root_vel = env.scene["robot"].data.root_com_vel_w
    vel_exploded = torch.any(torch.abs(root_vel) > threshold, dim=-1)
    vel_nan = torch.any(torch.isnan(root_vel) | torch.isinf(root_vel), dim=-1)

    # Combine the masks
    reset_mask = action_exploded | action_nan | vel_exploded | vel_nan
    
    return reset_mask

def illegal_ground_contact(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces_with_ground = contact_sensor.data.force_matrix_w[
        :, sensor_cfg.body_ids, ...
    ].squeeze()
    # check if any contact force with the ground exceeds the threshold
    return (
        torch.max(torch.norm(contact_forces_with_ground, dim=-1), dim=1)[0] > threshold
    )
