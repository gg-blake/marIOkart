from __future__ import annotations
from desmume.emulator import DeSmuME
from src.visualization.draw import draw_paragraph, draw_points, draw_text, draw_triangles, draw_lines
from src.utils.vector import interpolate
import torch
import numpy as np
from src.core.memory import *
from src.utils.vector import project_to_screen as _project_to_screen
import torch
from typing import TypeAlias, Union
from functools import wraps
from torch._prims_common import DeviceLikeType

AVAILABLE_OVERLAYS: list[Callable[[DeSmuME, DeviceLikeType | None], None]] = []


def register_overlay(func: Callable[[DeSmuME, DeviceLikeType | None], None]):
    AVAILABLE_OVERLAYS.append(func)
    return func

""" Display Collision Triangles in Overlay """

@register_overlay
def collision_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):

    kcl = load_current_kcl(emu, device=device)
    position = read_position(emu, device=device)
    indices = kcl.search_triangles(position)
    if indices is None or len(indices) == 0:
        return

    indices = torch.tensor(indices, dtype=torch.int32, device=device)
    triangles = kcl.triangles
    color_map = [
        (kcl.prisms.is_wall, lambda x: x == 1, (1, 0, 1)),
        # (racer.kcl.prisms.is_floor, lambda x: x == 1, (0, 0.5, 1)),
        (
            kcl.prisms.collision_type,
            lambda x: ((x == 3) | (x == 2) | (x == 5)),
            (1, 0, 0.3),
        ),
    ]

    for attr, cond, color in color_map:
        # filter triangles by attribute condition
        condition_mask = cond(attr[indices])
        indices_masked = indices[condition_mask]
        if indices_masked.shape[0] == 0:
            continue

        # project triangles to screen space
        v1, v2, v3 = triangles[indices_masked].chunk(3, dim=1)
        v1_proj, mask1 = project_to_screen(emu, v1.squeeze(1), device=device)
        v2_proj, mask2 = project_to_screen(emu, v2.squeeze(1), device=device)
        v3_proj, mask3 = project_to_screen(emu, v3.squeeze(1), device=device)
        
        # clip z
        valid_mask = mask1 & mask2 & mask3
        if not valid_mask.all():
            continue
        
        v1_proj = torch.cat([v1_proj[:, :2], v1_proj[:, 3, None]], dim=-1)
        v2_proj = torch.cat([v2_proj[:, :2], v2_proj[:, 3, None]], dim=-1)
        v3_proj = torch.cat([v3_proj[:, :2], v3_proj[:, 3, None]], dim=-1)
        v1_np = v1_proj.detach().cpu().numpy()
        v2_np = v2_proj.detach().cpu().numpy()
        v3_np = v3_proj.detach().cpu().numpy()
        draw_triangles(v1_np, v2_np, v3_np, np.array(color))

    

""" Display Kart Raycasting """
current_point = None

@register_overlay
def raycasting_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):
    position = read_position(emu, device=device)
    dir_f = read_direction(emu, device=device)
    dir_f /= dir_f.norm(dim=-1, keepdim=True)
    dir_l = torch.cross(dir_f, torch.tensor([0, 1.0, 0], device=device))
    dir_l /= dir_l.norm(dim=-1, keepdim=True)
    dir_r = -dir_l
    
    def _overlay(dir: torch.Tensor, color: np.ndarray, **sample_kwargs):
        nonlocal position
        points_f = read_facing_point_obstacle(
            emu,
            dir, 
            device=device,
            **sample_kwargs
        )
        f: torch.Tensor = read_forward_distance_obstacle(emu, device=device, interval=(-0.1, 0.1), n_steps=24)
        l: torch.Tensor = read_left_distance_obstacle(emu, device=device, interval=(-0.1, 0.1), n_steps=24)
        r: torch.Tensor = read_right_distance_obstacle(emu, device=device, interval=(-0.1, 0.1), n_steps=24)
        #print(torch.cat([l, f, r], dim=-1))
        
        if points_f is None:
            return
            
        if points_f.shape[0] == 0:
            return
        
        raycasted_points_proj, _ = project_to_screen(emu, points_f, device=device)
    
        # depth filter 2
        if raycasted_points_proj.shape[0] == 0:
            return
    
        # display depth norm, preserve depth in 3d
        depth_norm = raycasted_points_proj[:, 3, None]
        depth = raycasted_points_proj[:, 2, None]
        raycasted_points_proj = torch.cat([raycasted_points_proj[:, :2], depth_norm, depth], dim=-1)
        raycasted_points_proj = raycasted_points_proj[:, :3]
        intersect_proj_np = raycasted_points_proj.detach().cpu().numpy()
        
        pos = position[None, :].repeat(raycasted_points_proj.shape[0], 1)
        pos_proj, _ = project_to_screen(emu, pos, device=device)
        pos_proj = pos_proj[:, :3]
        pos_proj[:, 2] = 0.1
        pos_proj_np = pos_proj.detach().cpu().numpy()
        
        draw_lines(pos_proj_np, intersect_proj_np, colors=color)
        
    _overlay(dir_f, color=np.array([0.5, 0.7, 0.9]), interval=(-0.1, 0.1), n_steps=24)
    _overlay(dir_l, color=np.array([0.9, 0.5, 0.7]), interval=(-0.1, 0.1), n_steps=24)
    _overlay(dir_r, color=np.array([0.3, 0.9, 0.5]), interval=(-0.1, 0.1), n_steps=24)

@register_overlay
def camera_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):
    global racer, current_point

    camera_target = read_camera_target_position(emu, device=device)
    points, _ = project_to_screen(emu, camera_target.unsqueeze(0), device=device)
    points_np = points.detach().cpu().numpy()
    draw_points(points_np, colors=np.array([1.0, 0.0, 0.0]), radius_scale=5.0)


""" Displays an overlay of a line connecting checkpoint endpoints of the next checkpoint. """

@register_overlay
def checkpoint_overlay_1(emu: DeSmuME, device: DeviceLikeType | None = None):
    global current_point
    position = read_position(emu, device=device)
    
    checkpoint = read_next_checkpoint_position(emu, device=device)
    checkpoint[:, 1] = position[1]
    
    checkpoint_proj, _ = project_to_screen(emu, checkpoint, device=device)
    if checkpoint_proj.shape[0] < 2:
        return

    # display depth norm, preserve depth in 3d
    depth_norm = checkpoint_proj[:, 3, None] / 3
    depth = checkpoint_proj[:, 2, None]
    checkpoint_proj = torch.cat([checkpoint_proj[:, :2], depth_norm, depth], dim=-1)
    p1_np, p2 = checkpoint_proj[:, :3].chunk(2, dim=0)
    p1_np = p1_np.detach().cpu().numpy()
    p2_np = p2.detach().cpu().numpy()
    draw_lines(p1_np, p2_np, colors=np.array([0.0, 1.0, 0.0]), stroke_width_scale=1.0)


""" Displays an overlay of a ray connecting the kart and the next checkpoint boundary. """

@register_overlay
def checkpoint_overlay_2(emu: DeSmuME, device: DeviceLikeType | None = None):
    position = read_position(emu, device=device)
    direction = read_direction(emu, device=device)
    intersect = read_facing_point_checkpoint(emu, direction, device=device)
    intersect = intersect.unsqueeze(0)
    intersect_proj, _ = project_to_screen(emu, intersect, device=device)
    if intersect_proj.shape[0] == 0:
        return

    # display depth norm, preserve depth in 3d
    depth_norm = intersect_proj[:, 3, None]
    depth = intersect_proj[:, 2, None]
    intersect_proj = torch.cat([intersect_proj[:, :2], depth_norm, depth], dim=-1)
    intersect_proj = intersect_proj[:, :3]
    intersect_proj_np = intersect_proj.detach().cpu().numpy()
    draw_points(intersect_proj_np, colors=np.array([0.0, 1.0, 0.0]), radius_scale=1.0)

    intersect_proj_np[0, 2] = 0.1
    pos_proj, _ = project_to_screen(emu, position.unsqueeze(0), device=device)
    pos_proj = pos_proj[:, :3]
    pos_proj[:, 2] = 0.1
    pos_proj_np = pos_proj.detach().cpu().numpy()
    
    draw_lines(intersect_proj_np, pos_proj_np, colors=np.array([0.0, 0.0, 1.0]), stroke_width_scale=1.0)

@register_overlay
def player_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):
    objects = read_objects(emu)

    objs = [[], [], [], []]

    colors = [(0.7, 0.1, 0.6), (0.1, 0.7, 0.6), (0.1, 0.6, 0.7), (0.6, 0.1, 0.7)]
    for i, (key, ids) in enumerate(objects.items()):
        positions = []
        for id in ids:
            positions.append(read_object_position(emu, id, device=device))

        if len(positions) == 0: continue
        positions = torch.stack(positions, dim=0)
        object_positions, _ = project_to_screen(emu, positions, device=device)

        
        
        if object_positions.shape[0] == 0:
            continue

        object_positions = torch.cat(
            [object_positions[:, :2], object_positions[:, 3, None]], dim=-1
        )

        object_positions_np = object_positions.detach().cpu().numpy()
        colors_np = np.array(colors[i])
        draw_points(object_positions_np, colors=colors_np, radius_scale=5.0)
        
@register_overlay
def distance_overlay(emu: DeSmuME, device: DeviceLikeType | None = None):
    # Computation #
    driver = read_driver(emu)
    pos = read_VecFx32(driver.position, device=device)
    mtx = read_MtxFx32(driver.mainMtx, device)
    right, _, fwd, _ = torch.chunk(mtx, 4, dim=0)   
    
    fwd_p = read_closest_obstacle_point(emu, fwd.squeeze(), device=device) # raycast forward
    right_p = read_closest_obstacle_point(emu, right.squeeze(), device=device) # raycast right
    left_p = read_closest_obstacle_point(emu, -right.squeeze(), device=device) # raycast left
    
    
    # Display Logic #
    pos_proj, _ = project_to_screen(emu, pos.unsqueeze(0), device=device)
    pos_proj = pos_proj[:, :3]
    
    
    colors = []
    stack = []
    if fwd_p is not None:
        stack.append(fwd_p)
        colors.append(torch.tensor([0.0, 0.0, 1.0], device=device))
    if right_p is not None:
        stack.append(right_p)
        colors.append(torch.tensor([0.0, 1.0, 0.0], device=device))
    if left_p is not None:
        stack.append(left_p)
        colors.append(torch.tensor([1.0, 0.0, 0.0], device=device))
        
    if len(stack) == 0:
        return
    
    p_proj, mask = project_to_screen(emu, torch.stack(stack, dim=0), device=device)
    p_proj = p_proj[:, :3]
    
    if p_proj.shape[0] == 0:
        return
    
    pos_proj = pos_proj.repeat(p_proj.shape[0], 1)
    pos_proj = pos_proj.detach().cpu().numpy()
    p_proj = p_proj.detach().cpu().numpy()
    
    colors = torch.stack(colors, dim=0)
    colors = colors[mask]
    colors = colors.detach().cpu().numpy()
    
    draw_lines(pos_proj, p_proj, colors)
    
    
    