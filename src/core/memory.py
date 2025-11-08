"""
Mario Kart DS (MKDS) Emulator I/O & Geometry Utilities
======================================================

This module provides a high-level, vectorized interface for reading game state
from a running DeSmuME emulator and performing common geometric operations
used in visualization, control, and RL policy features for Mario Kart DS.

It wraps low-level memory reads (positions, directions, camera data, objects,
checkpoints, clock) and exposes a compact API for tasks like projecting
world-space points to screen-space, computing distances to checkpoints and
obstacles, and deriving view matrices.

The implementation favors:
  * **Deterministic caching** at frame and game lifetimes to minimize emulator
    I/O, and
  * **Torch tensor** computations (CPU / CUDA / MPS) for fast, batched math.

-------------------------------------------------------------------------------
Quick Start
-------------------------------------------------------------------------------

>>> from desmume.emulator import DeSmuME
>>> import torch
>>> from your_module import (
...     read_position, read_direction, project_to_screen, read_forward_distance_checkpoint
... )
>>> emu = DeSmuME()
>>> # ... open ROM, load state, etc.
>>> device = torch.device("cpu")  # or "cuda", "mps"
>>> pos = read_position(emu, device=device)               # (3,)
>>> dir = read_direction(emu, device=device)              # (3,)
>>> screen = project_to_screen(emu, pos.unsqueeze(0), device=device)  # (1, 4)
>>> fwd_to_cp = read_forward_distance_checkpoint(emu, device=device)  # scalar tensor

-------------------------------------------------------------------------------
Key Concepts & Conventions
-------------------------------------------------------------------------------

Coordinate Systems
------------------
* **World Space**: Right-handed, with **Y as up**. Many functions assume a
  canonical "up" vector of (0, 1, 0) and construct a right-handed orthonormal
  basis `[right, up, forward]`.
* **Camera Space / Clip Space / NDC / Screen Space**: `_compute_model_view`
  builds a model-view matrix from camera position/target; `_project_to_screen`
  applies perspective and viewport transforms to return pixel coordinates for a
  256×192 screen (Nintendo DS top display).
* **Screen Origin**: (0, 0) is **top-left**. X grows to the right; Y grows
  downward. This follows the standard raster convention and matches the
  `(1 - ndc_y)` transform used in projection.

Units
-----
* **Positions & Scalars** returned from memory are derived from MKDS fixed-point
  formats (FX32, etc.) via helpers like `read_vector_3d_fx32` and `read_fx32`,
  and are exposed as Python floats / Torch tensors.
* **Angles**:
  - Camera FOV is read from a 16-bit fixed-point angle and converted to radians
    using: ``value * (2π / 0x10000)``.
* **Time**:
  - `read_clock()` returns **centiseconds** (10 ms units).

Memory Map & Assets
-------------------
* **Addresses**: The module uses static addresses for key pointers (racer,
  course, objects, checkpoints, camera, clock). See constants:
  `RACER_PTR_ADDR`, `COURSE_ID_ADDR`, `OBJECTS_PTR_ADDR`, `CHECKPOINT_PTR_ADDR`,
  `CLOCK_DATA_PTR`, `CAMERA_PTR_ADDR`.
* **Course Files**:
  - `courses.json` maps course IDs to directory names.
  - KCL (`course_collision.kcl`) and NKM (`course_map.nkm`) are loaded via
    `KCLTensor.from_file(...)` and `NKMTensor.from_file(...)`.

Caching Model
-------------
Two decorators reduce emulator I/O:

* ``@frame_cache`` — Caches the function's **single** return value per
  emulator tick (`emu.get_ticks()`). Recomputes only when the tick changes.

* ``@game_cache`` — Caches the function's **single** return value for the
  process lifetime (until interpreter exit).

⚠ **Important**: Both caches **ignore argument values**. If you call a cached
function with different arguments within the same lifetime (same frame or same
run), the **first computed result is reused**. In practice, pass stable
arguments (e.g., a constant `device`) to avoid surprises.

Device Handling
---------------
Many functions accept a Torch `device` and return tensors allocated there. For
best performance, use `cuda` (GPU) or `mps` (Apple Silicon) when available, and
keep devices **consistent** across the call sites, especially for `@game_cache`
results (KCL/NKM tensors are created on the device used at first call).

-------------------------------------------------------------------------------
Public API Overview
-------------------------------------------------------------------------------

Clock & Course
--------------
* `read_clock_ptr(emu)` — Base pointer to clock data (cached for game lifetime).
* `read_clock(emu)` — Current clock in 10 ms units (cached per frame).
* `get_current_course_id(emu)` — Current course ID (byte).
* `get_course_path(id)` — Course directory name from `courses.json`.
* `load_current_kcl(emu, device)` — Parsed KCL collision mesh (game-cached).
* `load_current_nkm(emu, device)` — Parsed NKM map data (game-cached).

Player & Objects
----------------
* `read_racer_ptr(emu)` — Pointer to the player racer struct.
* `read_position(emu, device)` — Player world position `(3,)`.
* `read_direction(emu, device)` — Player forward direction `(3,)`.
* `read_objects(...)`, `read_object_*` helpers — Scans and queries object table.
  - `safe_object` decorator returns `None` for deleted objects.

Camera & Projection
-------------------
* `read_camera_ptr(emu)` — Pointer to camera struct.
* `read_camera_fov(emu)` — FOV in radians.
* `read_camera_aspect(emu)` — Aspect ratio (W/H).
* `read_camera_position(emu, device)` — Camera world pos `(3,)` with elevation.
* `read_camera_target_position(emu, device)` — Camera look-at `(3,)`.
* `read_model_view(emu, device)` — 4×4 model-view matrix.
* `project_to_screen(emu, points, device)` — Projects `(N,3)` to `(N,4)`:
  `[x_px, y_px, clip_z, normalized_depth]`.
* `z_clip_mask(x)` — Mask for points within Z-near/far bounds (camera space).

Checkpoints
~~~~~~~~~~~
* `read_checkpoint_ptr(emu)` — Pointer to checkpoint manager.
* `read_current_checkpoint(emu)`, `read_current_key_checkpoint(emu)`,
  `read_current_lap(emu)` — Indices for current progress.
* `read_ghost_checkpoint(emu)`, `read_ghost_key_checkpoint(emu)` — Ghost state.
* `read_checkpoint_positions(emu, device)` — `(C, 2, 3)` segment endpoints.
* `read_next_checkpoint(emu, checkpoint_count)` — Next index (wraps).
* `read_next_checkpoint_position(emu, device)`,
  `read_current_checkpoint_position(emu, device)` — `(2,3)` endpoints.
* `read_facing_point_checkpoint(emu, direction, device)` — Intersection of a
  ray (from player, given direction) with next checkpoint line in XZ.
* `read_forward_distance_checkpoint(emu, device)`,
  `read_left_distance_checkpoint(emu, device)`,
  `read_direction_to_checkpoint(emu, device)` — Distances/steering angle.

Obstacles (Walls / Offroad)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* `read_facing_point_obstacle(emu, position, direction, device)` — Samples a
  cone of rays around the forward direction to find the **nearest** hit against
  wall/offroad triangles. Returns a point or `None`.
* `read_forward_distance_obstacle(emu, device)`,
  `read_left_distance_obstacle(emu, device)`,
  `read_right_distance_obstacle(emu, device)` — Scalar distances to nearest
  obstacles along canonical forward/left/right rays. Return `+inf` when no hit.

-------------------------------------------------------------------------------
Return Types & Shapes
-------------------------------------------------------------------------------

* Positions / Directions: `torch.Tensor` with shape `(3,)`.
* Batches of points: `(N, 3)`.
* Screen Projection: `(N, 4)` → `[x_px, y_px, clip_z, normalized_depth]`.
* Checkpoints: `(C, 2, 3)` → per checkpoint two endpoints `[p1, p2]`.
* Distances / Angles: 0-D or 1-D scalar `torch.Tensor` (depending on operation).

-------------------------------------------------------------------------------
Errors & Edge Cases
-------------------------------------------------------------------------------

* **Deleted / Ignored Objects**: `safe_object`-wrapped functions return `None`
  when the object is deleted; callers must handle `None`.
* **No Geometry**: When there are no wall/offroad triangles or raycasts miss,
  obstacle distance functions return `+inf` (as a tensor).
* **Empty Projections**: `_project_to_screen` returns an empty tensor when
  given no points; invalid (behind-camera) points may still project with
  negative `clip_w`.

-------------------------------------------------------------------------------
Performance Notes
-------------------------------------------------------------------------------

* Caching eliminates redundant memory reads across a frame / game run.
* Geometry routines (ray casting, distances, projection) are vectorized in
  Torch; prefer GPU/MPS devices when available.
* Keep devices consistent across calls that share cached state (e.g., KCL/NKM).

-------------------------------------------------------------------------------
Examples
-------------------------------------------------------------------------------

Project player and next checkpoint endpoints to screen:

>>> pts = torch.vstack([read_position(emu, device),    # (1,3)
...                     read_next_checkpoint_position(emu, device)]).reshape(-1, 3)
>>> screen_pts = project_to_screen(emu, pts, device)
>>> screen_pts[:, :2]  # pixel coordinates

Compute lateral vs forward distance to next checkpoint:

>>> d_left  = read_left_distance_checkpoint(emu, device)
>>> d_front = read_forward_distance_checkpoint(emu, device)

Find nearest obstacle straight ahead:

>>> d_obs = read_forward_distance_obstacle(emu, device)
>>> float(d_obs) if torch.isfinite(d_obs) else float("inf")

-------------------------------------------------------------------------------
Implementation Notes
-------------------------------------------------------------------------------

* `_compute_orthonormal_basis` builds a right-handed frame from a forward
  vector and an up-like reference (default `(0,1,0)`), normalizing each axis.
* `_compute_model_view` constructs a 4×4 model-view matrix in row-major with
  basis rows `[right, up, forward]` and a translated origin.
* `_project_to_screen` creates a simple perspective matrix using vertical FOV
  and aspect ratio; returns pixel coordinates using constants
  `SCREEN_WIDTH = 256` and `SCREEN_HEIGHT = 192`.

-------------------------------------------------------------------------------
Compatibility
-------------------------------------------------------------------------------

* Tested with DeSmuME Python bindings and Torch. Some ops may vary by backend
  (e.g., MPS lacks a few linear algebra kernels); this module sticks to widely
  supported APIs.

"""
from __future__ import annotations
import sys, os
from desmume.emulator import SCREEN_WIDTH, DeSmuME
import ctypes
from mkds.kcl import read_fx32
from torch._prims_common import DeviceLikeType
from src.mkds_extensions.kcl_torch import KCLTensor
from src.mkds_extensions.nkm_torch import NKMTensor
from mkds.utils import read_vector_3d_fx32
import torch
import json
import math
from src.utils.vector import (
    pairwise_distances_cross,
    intersect_ray_line_2d,
    sample_semicircular_sweep,
    triangle_raycast_batch,
    sample_cone,
    triangle_altitude,
)
from private.mkds import camera_t, driver_t, struct_VecFx32

from typing import Callable, Concatenate, TypeVar, ParamSpec
from functools import wraps

P = ParamSpec("P")
R = TypeVar("R")

SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192
Z_FAR = 1000.0
Z_NEAR = 0.0
Z_SCALE = 10.0

RACER_PTR_ADDR = 0x0217ACF8
COURSE_ID_ADDR = 0x23CDCD8
OBJECTS_PTR_ADDR = 0x0217B588
CHECKPOINT_PTR_ADDR = 0x021755FC
CLOCK_DATA_PTR = 0x0217AA34
CAMERA_PTR_ADDR = 0x0217AA4C

# Object flags
FLAG_DYNAMIC = 0x1000
FLAG_MAPOBJ = 0x2000
FLAG_ITEM = 0x4000
FLAG_RACER = 0x8000

def frame_cache(
    func: Callable[Concatenate[DeSmuME, P], R],
) -> Callable[Concatenate[DeSmuME, P], R]:
    """Decorator that caches a function's return value once per emulator tick.

    The wrapped function will only be re-executed when `emu.get_ticks()` changes.
    Useful for expensive reads that don't change within a single frame.

    Args:
        func: A function whose first argument is a `DeSmuME` instance.

    Returns:
        A wrapped function with identical signature that returns a cached result per tick.
    """
    val = None
    frame_count = 0
    @wraps(func)
    def wrapper(emu: DeSmuME, *args: P.args, **kwargs: P.kwargs) -> R:
        wrapper.__doc__ = func.__doc__
        nonlocal frame_count, val
        if emu.get_ticks() != frame_count or val is None:
            frame_count = emu.get_ticks()
            val = func(emu, *args, **kwargs)

        return val

    return wrapper


def game_cache(
    func: Callable[Concatenate[DeSmuME, P], R],
) -> Callable[Concatenate[DeSmuME, P], R]:
    """Decorator that caches a function's return value for the process lifetime.

    The wrapped function executes once and its result is reused thereafter.
    Appropriate for data that remains constant across a run (e.g., course files).

    Args:
        func: A function whose first argument is a `DeSmuME` instance.

    Returns:
        A wrapped function with identical signature that returns a cached result.
    """
    val = None
    @wraps(func)
    def wrapper(emu: DeSmuME, *args: P.args, **kwargs: P.kwargs) -> R:
        wrapper.__doc__ = func.__doc__
        nonlocal val
        if val is None:
            val = func(emu, *args, **kwargs)

        return val

    return wrapper


def z_clip_mask(x: torch.Tensor) -> torch.Tensor:
    """Compute a boolean mask for points within the view frustum Z range.

    Args:
        x: Tensor of shape (N, 3+) where x[:, 2] is the camera-space Z.

    Returns:
        A boolean tensor of shape (N,) where True indicates Z is between -Z_FAR and -Z_NEAR.
    """
    return (x[:, 2] < -Z_NEAR) & (x[:, 2] > -Z_FAR)


@game_cache
def read_clock_ptr(emu: DeSmuME):
    """Read the base pointer to the game's clock data structure.

    Args:
        emu: Emulator instance.

    Returns:
        Integer address of the clock data struct.
    """
    return emu.memory.unsigned.read_long(CLOCK_DATA_PTR)


@frame_cache
def read_clock(emu: DeSmuME):
    """Read the current game clock value.

    The value is read from the clock data structure and multiplied by 10,
    resulting in units of 10 ms (centiseconds).

    Args:
        emu: Emulator instance.

    Returns:
        Integer time in 10 ms units.
    """
    addr = read_clock_ptr(emu)
    return emu.memory.signed.read_long(addr + 0x08) * 10


def get_current_course_id(emu: DeSmuME):
    """Read the current course ID from memory.

    Args:
        emu: Emulator instance.

    Returns:
        Integer course ID (byte).
    """
    return emu.memory.unsigned.read_byte(COURSE_ID_ADDR)


def get_course_path(id: int, lookup_path: str = "./src/misc/courses.json"):
    """
    Resolve a course ID to the local filesystem path for its assets.

    Args:
        id: Course ID.

    Returns:
        String path relative to ./private/courses/ for the given course.

    Raises:
        AssertionError: If the course ID is not present in the lookup table.
    """
    course_id_lookup = None
    with open(lookup_path, "r") as f:
        course_id_lookup = json.load(f)

    assert course_id_lookup is not None
    assert str(id) in course_id_lookup
    return course_id_lookup[str(id)]


@game_cache
def load_current_kcl(emu: DeSmuME, device):
    """Load and parse the KCL collision file for the current course.

    Cached for the lifetime of the process.

    Args:
        emu: Emulator instance.
        device: Torch device (e.g., 'cpu', 'cuda', 'mps') to store tensors on.

    Returns:
        `KCLTensor` with triangle and prism data on the specified device.
    """
    assert device is not None
    id = get_current_course_id(emu)
    path = get_course_path(id)
    path = f"./private/courses/{path}/course_collision.kcl"
    kcl = KCLTensor.from_file(path, device=device)
    return kcl


def read_VecFx32(vec, device):
    return torch.tensor([
        vec.x,
        vec.y,
        vec.z
    ], device=device, dtype=torch.float32) / 0x1000
    
def read_MtxFx32(mtx, device):
    return torch.tensor(mtx.m, device=device, dtype=torch.float32) / 0x1000

@game_cache
def load_current_nkm(emu: DeSmuME, device):
    """Load and parse the NKM map file for the current course.

    Cached for the lifetime of the process.

    Args:
        emu: Emulator instance.
        device: Torch device to store tensors on.

    Returns:
        `NKMTensor` with NKM section tensors (e.g., checkpoints) on the specified device.
    """
    id = get_current_course_id(emu)
    path = get_course_path(id)
    path = f"./private/courses/{path}/course_map.nkm"
    nkm = NKMTensor.from_file(path, device=device)
    return nkm


def read_racer_ptr(emu: DeSmuME, addr: int = RACER_PTR_ADDR):
    """Read the pointer to the player's racer object.

    Args:
        emu: Emulator instance.
        addr: Memory address where the racer pointer is stored.

    Returns:
        Integer address of the racer structure.
    """
    return emu.memory.unsigned.read_long(addr)


@frame_cache
def read_driver(emu: DeSmuME) -> driver_t:
    addr = read_racer_ptr(emu)
    data = bytes(emu.memory.unsigned[addr: addr+ctypes.sizeof(driver_t)])
    driver = driver_t.from_buffer_copy(data)
    return driver


@frame_cache
def read_position(emu: DeSmuME, device):
    """Read the player's world-space position.

    Args:
        emu: Emulator instance.
        device: Torch device for the returned tensor.

    Returns:
        torch.Tensor of shape (3,) representing (x, y, z) in world units.
    """
    data = emu.memory.unsigned
    addr = read_racer_ptr(emu)
    pos = read_vector_3d_fx32(data, addr + 0x80)
    return torch.tensor(pos, dtype=torch.float32, device=device)


@frame_cache
def read_direction(emu: DeSmuME, device):
    """Read the player's forward direction vector (world-space).

    Args:
        emu: Emulator instance.
        device: Torch device for the returned tensor.

    Returns:
        torch.Tensor of shape (3,) representing the forward direction.
    """
    data = emu.memory.unsigned
    addr = read_racer_ptr(emu)
    pos = read_vector_3d_fx32(data, addr + 0x68)
    return torch.tensor(pos, dtype=torch.float32, device=device)


def read_objects_array_max_count(emu: DeSmuME, addr: int = OBJECTS_PTR_ADDR):
    """Read the maximum number of objects in the global object array.

    Args:
        emu: Emulator instance.
        addr: Base address of the object array metadata.

    Returns:
        Signed integer max count.
    """
    return emu.memory.signed.read_long(addr + 0x08)


def read_objects_array_ptr(emu: DeSmuME, addr: int = OBJECTS_PTR_ADDR):
    """Read the pointer to the global object pointer array.

    Args:
        emu: Emulator instance.
        addr: Base address of the object array metadata.

    Returns:
        Signed integer address of the object pointer array.
    """
    return emu.memory.signed.read_long(addr + 0x10)


def read_object_offset(emu: DeSmuME, id: int):
    """Compute the memory offset of an object entry within the array.

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        Integer byte offset to the object's metadata entry.
    """
    return read_objects_array_ptr(emu) + id * 0x1C


def read_object_ptr(emu: DeSmuME, id: int):
    """Read the object instance pointer for a given object ID.

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        Integer address of the object struct (0 if null).
    """
    offset = read_object_offset(emu, id)
    return emu.memory.unsigned.read_long(offset + 0x18)


def read_object_flags(emu: DeSmuME, id: int):
    """Read the object's flags (type/category bits, state, etc.).

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        Unsigned short flags value.
    """
    offset = read_object_offset(emu, id)
    return emu.memory.unsigned.read_short(offset + 0x14)


def read_object_position_ptr(emu: DeSmuME, id: int):
    """Read the pointer to an object's position vector in memory.

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        Integer address for the object's position struct (0 if deleted).
    """
    offset = read_object_offset(emu, id)
    return emu.memory.unsigned.read_long(offset + 0x0C)


def read_object_is_ignored(emu: DeSmuME, id: int):
    """Determine if an object should be ignored (null or ignored-flag set).

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        True if object ptr is 0 or ignored bit is set; False otherwise.
    """
    obj_ptr = read_object_ptr(emu, id)
    flags = read_object_flags(emu, id)
    return obj_ptr == 0 or flags & 0x200


def read_object_is_deleted(emu: DeSmuME, id: int):
    """Check if the object has been deleted (position pointer is null).

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        True if deleted; False otherwise.
    """
    pos_ptr = read_object_position_ptr(emu, id)
    return pos_ptr == 0


def safe_object(func):
    """Decorator that skips object reads when the object appears deleted.

    The wrapped function receives `(emu, id, *args, **kwargs)`. If the object
    is deleted (null position pointer), the wrapper returns `None`.
    """

    def wrapper(emu: DeSmuME, id: int, *args, **kwargs):
        """Internal wrapper used by `safe_object` to guard deleted objects."""
        if read_object_is_deleted(emu, id):
            return None

        return func(emu, id, *args, **kwargs)

    return wrapper


@frame_cache
@safe_object
def read_object_position(emu: DeSmuME, id: int, device):
    """Read an object's world-space position.

    Args:
        emu: Emulator instance.
        id: Object index.
        device: Torch device for the returned tensor.

    Returns:
        torch.Tensor of shape (3,) in world coordinates, or None if deleted.
    """
    pos_ptr = read_object_position_ptr(emu, id)
    pos = read_vector_3d_fx32(emu.memory.unsigned, pos_ptr)
    return torch.tensor(pos, device=device)


@frame_cache
@safe_object
def read_map_object_type_id(emu: DeSmuME, id: int):
    """Read a map object's type ID (e.g., coin, tree, etc.).

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        Signed short type ID, or None if object is deleted.
    """
    obj_ptr = read_object_ptr(emu, id)
    return emu.memory.signed.read_short(obj_ptr)


@frame_cache
@safe_object
def read_map_object_is_coin_collected(emu: DeSmuME, id: int):
    """Check if a coin-type map object has been collected.

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        True if collected; False otherwise; or None if object is deleted.
    """
    obj_ptr = read_object_ptr(emu, id)
    return emu.memory.unsigned.read_short(obj_ptr + 0x02) & 0x01 != 0


@frame_cache
@safe_object
def read_racer_object_is_ghost(emu: DeSmuME, id: int):
    """Check if a racer object is currently in ghost state.

    Args:
        emu: Emulator instance.
        id: Object index.

    Returns:
        True if ghosted; False otherwise; or None if object is deleted.
    """
    obj_ptr = read_object_ptr(emu, id)
    ghost_flag = emu.memory.unsigned.read_byte(obj_ptr + 0x7C)
    return ghost_flag & 0x04 != 0


@frame_cache
def read_objects(emu: DeSmuME):
    """Scan the global object table and group object indices by category.

    Categories:
      - 'map_objects'
      - 'racer_objects'
      - 'item_objects'
      - 'dynamic_objects'

    Returns:
        Dict[str, list[int]] mapping category name to list of indices.
    """
    obj_ids: dict[str, list[int]] = {
        "map_objects": [],
        "racer_objects": [],
        "item_objects": [],
        "dynamic_objects": [],
    }
    max_count = read_objects_array_max_count(emu)
    count = 0
    idx = 0
    while idx < 255 and count != max_count:
        if read_object_is_deleted(emu, idx):
            continue
        else:
            count += 1

        if read_object_is_ignored(emu, idx):
            continue

        flags = read_object_flags(emu, idx)
        if flags & FLAG_MAPOBJ != 0:
            obj_ids["map_objects"].append(idx)
        elif flags & FLAG_RACER != 0:
            obj_ids["racer_objects"].append(idx)
        elif flags & FLAG_ITEM != 0:
            obj_ids["item_objects"].append(idx)
        elif flags & FLAG_DYNAMIC == 0:
            obj_ids["dynamic_objects"].append(idx)

        idx += 1

    return obj_ids


@frame_cache
def read_camera_ptr(emu: DeSmuME, addr: int = CAMERA_PTR_ADDR):
    """Read the pointer to the active camera structure.

    Args:
        emu: Emulator instance.
        addr: Address where the camera pointer is stored.

    Returns:
        Integer address of the camera struct.
    """
    return emu.memory.unsigned.read_long(addr)


@frame_cache
def read_camera_fov(emu: DeSmuME):
    """Read the current camera field-of-view (radians).

    The FOV value is stored as a 16-bit fixed-point angle; it is converted to radians.

    Args:
        emu: Emulator instance.

    Returns:
        Floating-point FOV in radians.
    """
    addr = read_camera_ptr(emu)
    return emu.memory.unsigned.read_short(addr + 0x60) * (2 * math.pi / 0x10000)


@frame_cache
def read_camera_aspect(emu: DeSmuME):
    """Read the camera aspect ratio from memory.

    Args:
        emu: Emulator instance.

    Returns:
        Float aspect ratio (width/height).
    """
    addr = read_camera_ptr(emu)
    return read_fx32(emu.memory.unsigned, addr + 0x6C)


@frame_cache
def read_camera_position(emu: DeSmuME, device):
    """Read the camera world position, including elevation offset.

    Args:
        emu: Emulator instance.
        device: Torch device for the returned tensor.

    Returns:
        torch.Tensor shape (3,) representing camera (x, y, z).
    """
    addr = read_camera_ptr(emu)
    pos = read_vector_3d_fx32(emu.memory.unsigned, addr + 0x24)
    elevation = read_fx32(emu.memory.unsigned, addr + 0x178)
    pos = (pos[0], pos[1] + elevation, pos[2])
    return torch.tensor(pos, device=device)


def read_camera_target_position(emu: DeSmuME, device):
    """Read the camera's target/look-at position in world space.

    Args:
        emu: Emulator instance.
        device: Torch device for the returned tensor.

    Returns:
        torch.Tensor shape (3,) target (x, y, z).
    """
    addr = read_camera_ptr(emu)
    pos = read_vector_3d_fx32(emu.memory.unsigned, addr + 0x18)
    return torch.tensor(pos, device=device)


def _compute_orthonormal_basis(
    forward_vector_3d: torch.Tensor,
    reference_vector_3d: torch.Tensor | None = None,
    device=None,
):
    """Compute a right-handed orthonormal basis given a forward vector.

    Args:
        forward_vector_3d: Tensor shape (3,) forward direction.
        reference_vector_3d: Optional up-like reference; defaults to (0,1,0).
        device: Unused (kept for signature parity).

    Returns:
        torch.Tensor shape (3,3) with rows [right, up, forward].
    """
    if reference_vector_3d is None:
        reference_vector_3d = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=forward_vector_3d.dtype,
            device=forward_vector_3d.device,
        )

    right_vector_3d = torch.cross(forward_vector_3d, reference_vector_3d, dim=0)
    right_vector_3d /= right_vector_3d.norm()

    up_vector_3d = torch.cross(right_vector_3d, forward_vector_3d, dim=0)
    up_vector_3d /= up_vector_3d.norm()

    basis = torch.stack(
        [
            right_vector_3d,
            up_vector_3d,
            forward_vector_3d,
        ],
        dim=0,
    )

    return basis


def _compute_model_view(
    camera_pos: torch.Tensor, camera_target_pos: torch.Tensor, device
):
    """Build a 4x4 model-view matrix from camera position and target.

    Args:
        camera_pos: Tensor shape (3,) camera world position.
        camera_target_pos: Tensor shape (3,) target look-at position.
        device: Torch device for the returned matrix.

    Returns:
        torch.Tensor shape (4,4) model-view matrix.
    """
    forward = camera_target_pos - camera_pos
    forward /= torch.norm(forward, dim=-1)

    rot = _compute_orthonormal_basis(forward, device=device)

    pos_proj = rot @ camera_pos.unsqueeze(-2).transpose(-1, -2)

    model_view = torch.eye(4, dtype=rot.dtype, device=device)
    model_view[:3, :3] = rot
    model_view[:3, 3] = -pos_proj.squeeze(-1)

    return model_view
  

@frame_cache
def read_camera(emu: DeSmuME) -> camera_t:
    addr = read_camera_ptr(emu)
    data = bytes(emu.memory.unsigned[addr: addr+ctypes.sizeof(camera_t)])
    camera = camera_t.from_buffer_copy(data)
    return camera
    

@frame_cache
def read_model_view(emu: DeSmuME, device):
    """Compute and cache the camera model-view matrix for the current frame.

    Args:
        emu: Emulator instance.
        device: Torch device for returned matrix.

    Returns:
        torch.Tensor shape (4,4) model-view matrix.
    """
    addr = read_camera_ptr(emu)
    
    # Load the camera struct
    data = bytes(emu.memory.unsigned[addr: addr+ctypes.sizeof(camera_t)])
    camera = camera_t.from_buffer_copy(data)
    mat = torch.tensor(camera.mtx.m, device=device)
    mat = torch.cat([mat.T, torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float)[None, :]], dim=0)
    
    # Move the decimal point over for fixed point data (fx32)
    mat[:3, :3] /= 0x1000
    mat[:3, 3] /= 0x100 # position is scaled by 16
    
    return mat

@game_cache
def read_projection(emu: DeSmuME, device):
    camera = read_camera(emu)

    fov_sin = camera.fovSin / 0x1000
    fov_cos = camera.fovCos / 0x1000
    aspect = camera.aspectRatio / 0x1000
    far = camera.frustumFar / 0x1000
    near = camera.frustumNear / 0x1000

    pm = torch.zeros((4, 4), device=device)
    pm[0, 0] = fov_cos / (fov_sin * aspect)
    pm[1, 1] = fov_cos / fov_sin
    pm[2, 2] = -(far + near) / (far - near)
    pm[2, 3] = -(2 * far * near) / (far - near)
    pm[3, 2] = -1
    
    return pm
  
    
def to_screen(emu: DeSmuME, points: torch.Tensor, screen_w, screen_h, device=None):
    mvm = read_model_view(emu, device)
    pm = read_projection(emu, device)
    camera = read_camera(emu)
    far = camera.frustumFar / 0x1000
    near = camera.frustumNear / 0x1000
    
    # convert to camera space
    padded = torch.nn.functional.pad(points, (0, 1), "constant", 1)
    cam_space = (mvm @ padded.T).T
    
    # convert to clip space
    clip_space = (pm @ cam_space.T).T
    
    # depth
    ndc = clip_space[:, :3] / clip_space[:, 3, None] # normalize w/ respect to w (new shape: (B, 3))
    
    # screen space
    screen_x = (ndc[:, 0] + 1) / 2 * screen_w
    screen_y = (1 - ndc[:, 1]) / 2 * screen_h
    screen_depth = clip_space[:, 2]
    screen_depth_norm = -far / (-far + 16 * clip_space[:, 2])
    screen_space = torch.stack([
        screen_x, 
        screen_y, 
        screen_depth, 
        screen_depth_norm
    ], dim=-1)
    
    z_clip = (screen_space[:, 2] > near) & (screen_space[:, 2] < far)
    
    out = screen_space[z_clip]
    return out, z_clip


def _project_to_screen(world_points, model_view, fov, aspect, screen_dim: tuple[int, int], device=None):
    """Project world-space points to screen coordinates using perspective projection.

    Args:
        world_points: Tensor shape (N,3) of world-space points.
        model_view: Tensor shape (4,4) model-view matrix.
        fov: Field-of-view in radians (vertical half-angle usage within projection).
        aspect: Aspect ratio (width/height).
        device: Torch device for intermediate/return tensors.

    Returns:
        Tensor shape (N,4): [x_px, y_px, clip_z, normalized_depth],
        where x/y are in pixel coordinates for a SCREEN_WIDTH x SCREEN_HEIGHT viewport.
    """
    N = world_points.shape[0]

    # Homogenize points
    ones = torch.ones((N, 1), device=device)
    world_points = torch.cat([world_points, ones], dim=-1)
    cam_space = (model_view @ world_points.T).T

    # Perspective projection
    f = torch.tan(torch.tensor(fov, device=device) / 2)

    if cam_space.shape[0] == 0:
        return torch.empty((0, 2), device=device)

    fov_h = math.tan(fov)
    fov_w = math.tan(fov) * aspect

    projection_matrix = torch.zeros((4, 4), device=device)
    projection_matrix[0, 0] = 1 / fov_w
    projection_matrix[1, 1] = 1 / fov_h
    projection_matrix[2, 2] = (Z_FAR + Z_NEAR) / (Z_NEAR - Z_FAR)
    projection_matrix[2, 3] = -(2 * Z_FAR * Z_NEAR) / (Z_NEAR - Z_FAR)
    projection_matrix[3, 2] = 1
    clip_space = (projection_matrix @ cam_space.T).T

    ndc = clip_space[:, :3] / clip_space[:, 3, None]

    screen_width, screen_height = screen_dim
    screen_x = (ndc[:, 0] + 1) / 2 * screen_width
    screen_y = (1 - ndc[:, 1]) / 2 * screen_height
    screen_depth = clip_space[:, 2]
    screen_depth_norm = -Z_FAR / (-Z_FAR + Z_SCALE * clip_space[:, 2])
    return torch.stack([screen_x, screen_y, screen_depth, screen_depth_norm], dim=-1)


def project_to_screen(emu: DeSmuME, points: torch.Tensor, device, screen_dim=(SCREEN_WIDTH, SCREEN_HEIGHT)):
    """Convenience wrapper to project points using the current camera state.

    Args:
        emu: Emulator instance.
        points: Tensor shape (N,3) of world-space points.
        device: Torch device.

    Returns:
        Tensor shape (N,4) in screen space (see `_project_to_screen`).
    """
    model_view = read_model_view(emu, device=device)
    fov = read_camera_fov(emu)
    aspect = read_camera_aspect(emu)
    return to_screen(emu, points, *screen_dim, device=device)

# CHECKPOINT INFO #


@game_cache
def read_checkpoint_ptr(emu: DeSmuME, addr: int = CHECKPOINT_PTR_ADDR):
    """Read the pointer to the checkpoint manager/state.

    Args:
        emu: Emulator instance.
        addr: Address where the checkpoint pointer is stored.

    Returns:
        Integer address for checkpoint data.
    """
    return emu.memory.unsigned.read_long(addr)


@frame_cache
def read_current_checkpoint(emu: DeSmuME):
    """Read the index of the current checkpoint.

    Args:
        emu: Emulator instance.

    Returns:
        Unsigned byte checkpoint index.
    """
    addr = read_checkpoint_ptr(emu)
    return emu.memory.unsigned.read_byte(addr + 0x46)


@frame_cache
def read_current_key_checkpoint(emu: DeSmuME):
    """Read the current key checkpoint index (special/lap-related).

    Args:
        emu: Emulator instance.

    Returns:
        Signed byte key checkpoint index.
    """
    addr = read_checkpoint_ptr(emu)
    return emu.memory.signed.read_byte(addr + 0x48)


@frame_cache
def read_ghost_checkpoint(emu: DeSmuME):
    """Read the recorded ghost's current checkpoint index.

    Args:
        emu: Emulator instance.

    Returns:
        Signed byte ghost checkpoint index.
    """
    addr = read_checkpoint_ptr(emu)
    return emu.memory.signed.read_byte(addr + 0xD2)


@frame_cache
def read_ghost_key_checkpoint(emu: DeSmuME):
    """Read the recorded ghost's current key checkpoint index.

    Args:
        emu: Emulator instance.

    Returns:
        Signed byte ghost key checkpoint index.
    """
    addr = read_checkpoint_ptr(emu)
    return emu.memory.signed.read_byte(addr + 0xD4)


@frame_cache
def read_current_lap(emu: DeSmuME):
    """Read the current lap number.

    Args:
        emu: Emulator instance.

    Returns:
        Signed byte lap index (0-based).
    """
    addr = read_checkpoint_ptr(emu)
    return emu.memory.signed.read_byte(addr + 0x38)


@frame_cache
def read_next_checkpoint(emu: DeSmuME, checkpoint_count: int):
    """Compute the next checkpoint index (wrapping to 0 at the end).

    Args:
        emu: Emulator instance.
        checkpoint_count: Total number of checkpoints.

    Returns:
        Integer index of the next checkpoint.
    """
    current_checkpoint = read_current_checkpoint(emu)
    next_checkpoint = current_checkpoint + 1
    if next_checkpoint != checkpoint_count:
        return next_checkpoint
    else:
        return 0

@frame_cache
def read_previous_checkpoint(emu: DeSmuME, checkpoint_count: int):
    current_checkpoint = read_current_checkpoint(emu)
    prev_checkpoint = current_checkpoint - 1
    if prev_checkpoint >= 0:
        return prev_checkpoint
    else:
        return checkpoint_count - 1

def _convert_2d_checkpoints(P: torch.Tensor, source: torch.Tensor, dim=0):
    """Lift 2D checkpoint endpoints into 3D by sampling the missing dimension.

    Given 2D endpoints (e.g., XZ) and a set of floor points, fills in the
    missing axis by nearest-neighbor on that axis.

    Args:
        P: Tensor shape (N,2) endpoints (with one missing dimension).
        source: Tensor shape (M,3) of reference points (e.g., floor vertices).
        dim: Dimension index (0/1/2) to fill in.

    Returns:
        Tensor shape (N,3) with the missing axis populated.
    """
    dim_mask = torch.range(0, source.shape[1] - 1, 1) != dim

    D = pairwise_distances_cross(P, source[:, dim_mask])
    min_idx = D.argmin(dim=1)

    result = torch.ones(P.shape[0], P.shape[1] + 1, device=P.device)
    result[:, dim_mask] = P
    result[:, dim] = source[min_idx, dim]

    return result


@game_cache
def read_checkpoint_positions(emu: DeSmuME, device):
    """Build a tensor of checkpoint segment endpoints in 3D.

    Reads NKM and KCL, extracts floor geometry, and converts checkpoint pairs
    from 2D to 3D using nearest floor elevation.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Tensor shape (C, 2, 3) where C is number of checkpoints,
        containing [p1, p2] endpoints per checkpoint.
    """
    nkm = load_current_nkm(emu, device=device)
    kcl = load_current_kcl(emu, device=device)
    floor_mask = kcl.prisms.is_floor == 1
    floor_points = kcl.triangles[floor_mask]
    floor_points = floor_points.reshape(floor_points.shape[0] * 3, 3)
    return torch.stack(
        [
            _convert_2d_checkpoints(nkm._CPOI.position1, floor_points, dim=1),
            _convert_2d_checkpoints(nkm._CPOI.position2, floor_points, dim=1),
        ],
        dim=1,
    )


@frame_cache
def read_next_checkpoint_position(emu: DeSmuME, device):
    """Get the 3D endpoints of the next checkpoint segment.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Tensor shape (2,3) representing the next checkpoint's [p1, p2].
    """
    nkm = load_current_nkm(emu, device=device)
    checkpoints = read_checkpoint_positions(emu, device)
    checkpoint_count = nkm._CPOI.entry_count
    next_checkpoint = read_next_checkpoint(emu, checkpoint_count)
    return checkpoints[next_checkpoint]
    
@frame_cache
def read_previous_checkpoint_position(emu: DeSmuME, device):
    """Get the 3D endpoints of the next checkpoint segment.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Tensor shape (2,3) representing the next checkpoint's [p1, p2].
    """
    nkm = load_current_nkm(emu, device=device)
    checkpoints = read_checkpoint_positions(emu, device)
    checkpoint_count = nkm._CPOI.entry_count
    previous_checkpoint = read_previous_checkpoint(emu, checkpoint_count)
    return checkpoints[previous_checkpoint]


@frame_cache
def read_current_checkpoint_position(emu: DeSmuME, device):
    """Get the 3D endpoints of the current checkpoint segment.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Tensor shape (2,3) representing current checkpoint's [p1, p2].
    """
    checkpoints = read_checkpoint_positions(emu, device=device)
    current_checkpoint = read_current_checkpoint(emu)
    return checkpoints[current_checkpoint]


@frame_cache
def read_facing_point_checkpoint(emu: DeSmuME, direction: torch.Tensor, device):
    """Raycast from the player along a direction to the next checkpoint line (XZ).

    Args:
        emu: Emulator instance.
        direction: Tensor shape (3,) direction vector.
        device: Torch device.

    Returns:
        Tensor shape (3,) point of intersection in world coordinates.
    """
    position = read_position(emu, device=device)
    checkpoint = read_next_checkpoint_position(emu, device)
    mask_xz = torch.tensor([0, 2], dtype=torch.int32, device=device)
    pos_xz = position[mask_xz]
    dir_xz = direction[mask_xz]
    pxz_1, pxz_2 = checkpoint[:, mask_xz].chunk(2, dim=0)
    pxz_1 = pxz_1.squeeze(0)
    pxz_2 = pxz_2.squeeze(0)
    intersect, _ = intersect_ray_line_2d(pos_xz, dir_xz, pxz_1, pxz_2)
    intersect = torch.tensor([intersect[0], position[1], intersect[1]], device=device)
    return intersect


@frame_cache
def read_forward_distance_checkpoint(emu, device):
    """Compute forward distance from player to the next checkpoint line.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Scalar torch.Tensor distance.
    """
    direction = read_direction(emu, device=device)
    position = read_position(emu, device=device)
    ray_point = read_facing_point_checkpoint(emu, direction, device=device)
    return torch.norm(ray_point - position)


@frame_cache
def read_left_distance_checkpoint(emu, device):
    """Compute leftward distance from player to the next checkpoint line.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Scalar torch.Tensor distance.
    """
    direction = read_direction(emu, device=device)
    position = read_position(emu, device=device)
    up_basis = -torch.tensor([0, 1.0, 0], device=device, dtype=torch.float32)
    left_basis = torch.cross(direction, up_basis)
    ray_point = read_facing_point_checkpoint(emu, left_basis, device=device)
    return torch.norm(ray_point - position)


@frame_cache
def read_direction_to_checkpoint(emu: DeSmuME, device):
    """Compute a steering angle toward the next checkpoint from forward/left distances.

    Angle is computed as atan(forward / left).

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Scalar torch.Tensor angle in radians.
    """
    f = read_forward_distance_checkpoint(emu, device=device)
    l = read_left_distance_checkpoint(emu, device=device)
    angle = torch.atan(f / l)
    return angle


# OBSTACLE INFO #


@frame_cache
def read_facing_point_obstacle(
    emu: DeSmuME,
    direction: torch.Tensor,
    device=None,
    **sample_kwargs
):
    """Raycast toward walls/offroad and return the nearest hit point.

    Samples a cone of directions around the provided (or player) direction, and
    finds the nearest intersection against wall and offroad triangles.

    Args:
        emu: Emulator instance.
        position: Optional world position (3,). Defaults to player's position.
        direction: Optional direction (3,). Defaults to player's forward vector.
        device: Torch device.

    Returns:
        torch.Tensor shape (3,) hit point, or None if no intersections.
    """
    assert device is not None
    kcl = load_current_kcl(emu, device=device)
    triangles = kcl.triangles
    wall_mask = (kcl.prisms.is_wall == 1
        | (kcl.prisms.collision_type == 8)
        | (kcl.prisms.collision_type == 9)
        | (kcl.prisms.collision_type == 16)
        | (kcl.prisms.collision_type == 21)
        | (kcl.prisms.collision_type == 14)
    )
    offroad_mask = (
        (kcl.prisms.collision_type == 5)
        | (kcl.prisms.collision_type == 3)
        | (kcl.prisms.collision_type == 2)
        
    )
    triangles = triangles[wall_mask | offroad_mask]
    B = triangles.shape[0]

    if B == 0:
        return None

    v1, v2, v3 = triangles.chunk(3, dim=1)
    v1 = v1.squeeze(1)
    v2 = v2.squeeze(1)
    v3 = v3.squeeze(1)


    driver = read_driver(emu)
    position = read_VecFx32(driver.position, device=device)
    up_vector = read_VecFx32(driver.upDir, device=device)
    left_vector = torch.cross(direction, up_vector)
    ray_samples = sample_semicircular_sweep(direction, left_vector, up_vector, **sample_kwargs)
    ray_dir = direction.reshape(1, 3)
    ray_samples = ray_dir#torch.cat([ray_dir, ray_samples], dim=0)

    ray_origin = position
    ray_origin = ray_origin.unsqueeze(0)
    ray_origin = ray_origin.reshape(1, 3)
    ray_origin_samples = ray_origin.repeat(ray_samples.shape[0], 1)

    points = triangle_raycast_batch(ray_origin_samples, ray_samples, v1, v2, v3)
    N, M, C = points.shape
    points = points.reshape(N * M, C)
    
    return points
    
    

@frame_cache
def read_closest_obstacle_point(
    emu: DeSmuME,
    direction: torch.Tensor,
    device = None,
    **sample_kwargs
) -> torch.Tensor | None:
    points = read_facing_point_obstacle(emu, direction, device=device, **sample_kwargs)
    
    if points is None:
        return None
    
    if points.shape[0] == 0:
        return None
    
    driver = read_driver(emu)
    position = read_VecFx32(driver.position, device=device)
    dist = torch.cdist(points, position.unsqueeze(0))
    min_id = torch.argmin(dist)
    current_point_min = points[min_id]
    return current_point_min

@frame_cache
def read_forward_distance_obstacle(emu: DeSmuME, device=None, **sample_kwargs) -> torch.Tensor:
    """Compute forward distance to the nearest wall/offroad obstacle.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Scalar torch.Tensor distance; +inf if no hit.
    """
    position = read_position(emu, device=device)
    direction = read_direction(emu, device)
    ray_point = read_closest_obstacle_point(emu, direction, device=device, **sample_kwargs)
    if ray_point is None:
        return torch.tensor([float("inf")], device=device)

    dist = torch.sqrt(torch.sum((position - ray_point) ** 2, dim=0, keepdim=True))
    
    return dist


@frame_cache
def read_left_distance_obstacle(emu: DeSmuME, device=None, **sample_kwargs) -> torch.Tensor:
    """Compute leftward distance to the nearest wall/offroad obstacle.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Scalar torch.Tensor distance; +inf if no hit.
    """
    position = read_position(emu, device=device)
    direction = read_direction(emu, device=device)
    up_basis = -torch.tensor([0, 1.0, 0], device=device, dtype=torch.float32)
    left_basis = torch.cross(direction, up_basis)
    ray_point = read_closest_obstacle_point(emu, direction=left_basis, device=device, **sample_kwargs)
    if ray_point is None:
        return torch.tensor([float("inf")], device=device)

    dist = torch.sqrt(torch.sum((position - ray_point) ** 2, dim=0, keepdim=True))
    return dist


@frame_cache
def read_right_distance_obstacle(emu: DeSmuME, device=None, **sample_kwargs) -> torch.Tensor:
    """Compute rightward distance to the nearest wall/offroad obstacle.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Scalar torch.Tensor distance; +inf if no hit.
    """
    position = read_position(emu, device=device)
    direction = read_direction(emu, device=device)
    up_basis = torch.tensor([0, 1.0, 0], device=device, dtype=torch.float32)
    right_basis = torch.cross(direction, up_basis)
    ray_point = read_closest_obstacle_point(emu, direction=right_basis, device=device, **sample_kwargs)
    if ray_point is None:
        return torch.tensor([float("inf")], device=device)

    dist = torch.sqrt(torch.sum((position - ray_point) ** 2, dim=0, keepdim=True))
    return dist


@frame_cache
def read_checkpoint_distance_altitude(emu: DeSmuME, device) -> torch.Tensor:
    """Compute the altitude (height) of the triangle formed by player and checkpoint endpoints.

    Uses the two checkpoint endpoints and the player's position to form sides a and b,
    then returns the triangle altitude via `triangle_altitude(a, b)`.

    Args:
        emu: Emulator instance.
        device: Torch device.

    Returns:
        Scalar torch.Tensor altitude value.
    """
    next_checkpoint = read_next_checkpoint_position(emu, device=device)
    p1, p2 = next_checkpoint.chunk(2, dim=0)

    position = read_position(emu, device=device)
    a = torch.norm(p1 - position)
    b = torch.norm(p2 - position)
    return triangle_altitude(a, b)


@frame_cache
def read_touching_prism_type(emu: DeSmuME, attr_mask: Callable[[torch.Tensor], torch.Tensor], device) -> bool:
    kcl = load_current_kcl(emu, device=device)
    position = read_position(emu, device=device)
    indices = kcl.search_triangles(position)
    if indices is None or len(indices) == 0:
        return False

    indices = torch.tensor(indices, dtype=torch.int32, device=device)
    
    triangle_attr = kcl.prisms.collision_type[indices]
    
    mask = attr_mask(triangle_attr)
    offroad_indices = indices[mask]
    return offroad_indices.shape[0] > 0
    
def read_mat_c(emu: DeSmuME, device = None):
    addr = read_camera_ptr(emu)
    data = bytes(emu.memory.unsigned[addr: addr+ctypes.sizeof(camera_t)])
    camera = camera_t.from_buffer_copy(data)
    mat = camera.mtx
    return torch.tensor(mat.m, device=device) / 0x1000
    
def read_pos_c(emu: DeSmuME, device = None):
    addr = read_camera_ptr(emu)
    data = bytes(emu.memory.unsigned[addr: addr+ctypes.sizeof(camera_t)])
    camera = camera_t.from_buffer_copy(data)
    pos = camera.position
    return torch.tensor([pos.x, pos.y, pos.z], device=device) / 0x1000
    
def read_driver_pos_c(emu: DeSmuME, device = None):
    addr = read_racer_ptr(emu)
    data = bytes(emu.memory.unsigned[addr: addr+ctypes.sizeof(driver_t)])
    driver = driver_t.from_buffer_copy(data)
    pos = driver.position
    return torch.tensor([pos.x, pos.y, pos.z], device=device) / 0x1000
    
