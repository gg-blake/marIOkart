# MKDS Emulator I/O & Geometry Utilities

> **Mario Kart DS (MKDS) emulator-memory helpers + vectorized geometry utilities for visualization, control, and RL.**  
> Read live game state from DeSmuME, project world points to screen space, and compute distances to checkpoints and obstacles — all with PyTorch tensors and deterministic caching.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Design & Conventions](#design--conventions)
  - [Coordinate Systems](#coordinate-systems)
  - [Units](#units)
  - [Memory Map & Assets](#memory-map--assets)
  - [Caching Model](#caching-model)
  - [Device Handling](#device-handling)
- [Constants](#constants)
- [API Reference](#api-reference)
  - [Decorators](#decorators)
  - [Utility](#utility)
  - [Clock & Course](#clock--course)
  - [Player & Objects](#player--objects)
  - [Camera & Projection](#camera--projection)
  - [Checkpoints](#checkpoints)
  - [Obstacles (Walls / Offroad)](#obstacles-walls--offroad)
  - [Internal Helpers](#internal-helpers)
- [Examples](#examples)
  - [Project the Player to Screen Space](#project-the-player-to-screen-space)
  - [Distances to Next Checkpoint](#distances-to-next-checkpoint)
  - [Nearest Obstacle Ahead](#nearest-obstacle-ahead)
- [Edge Cases & Error Handling](#edge-cases--error-handling)
- [Performance Notes](#performance-notes)
- [FAQ](#faq)
- [License](#license)



---

(overview)=
## Overview

This library provides a high-level, vectorized interface for reading Mario Kart DS game state from a running **DeSmuME** emulator and for performing common **3D/2D geometry** operations used in overlays, analytics, and reinforcement learning.

It wraps low-level memory reads (positions, directions, camera parameters, objects, checkpoints, clock) and exposes a compact API for tasks like:

- Projecting world-space points to screen-space (Nintendo DS 256×192),
- Computing distances and steering cues to the **next checkpoint**,
- Finding **nearest obstacles** (walls/offroad) via batched ray casting,
- Efficient, deterministic **caching** of memory-fetches per-frame and per-run.

The implementation favors **Torch tensors** (CPU / CUDA / MPS) for fast batched math and provides explicit control over devices.

---

(installation)=
## Installation

```bash
# Using a virtual environment is recommended
pip install -r requirements.txt
```

> You’ll also need the DeSmuME Python bindings and project-local modules that define `KCLTensor` and `NKMTensor` (collision and map parsers).

---

(getting-started)=
## Getting Started

```python
from desmume.emulator import DeSmuME
import torch

# Import what you need from the module that contains this library
from utils.memory import (
    read_position, read_direction, project_to_screen,
    read_forward_distance_checkpoint,
)

emu = DeSmuME()
# emu.open("mariokart_ds.nds"); emu.savestate.load(2); emu.volume_set(0)  # example lifecycle

device = torch.device("cpu")  # or "cuda", "mps"

pos = read_position(emu, device=device)                 # (3,)
dir = read_direction(emu, device=device)                # (3,)
screen = project_to_screen(emu, pos.unsqueeze(0), device=device)  # (1,4)
fwd_to_cp = read_forward_distance_checkpoint(emu, device=device)  # scalar tensor
```

---

(design--conventions)=
## Design & Conventions

(coordinate-systems)=
### Coordinate Systems

- **World Space**: Right-handed with **Y up**. Many functions assume `(0, 1, 0)` as the up-like reference.
- **Camera Space → Clip → NDC → Screen**: `read_model_view` builds the model-view matrix; `_project_to_screen` applies perspective projection and viewport mapping to DS resolution.
- **Screen Origin**: `(0, 0)` is **top-left**. X to the right, Y down.

(units)=
### Units

- **Positions/Scalars**: Converted from MKDS fixed-point memory formats (e.g., fx32) via helpers like `read_vector_3d_fx32` and `read_fx32`.
- **Angles**: Camera FOV is read as 16-bit fixed-point and converted using `value * (2π / 0x10000)` → radians.
- **Time**: `read_clock()` returns **centiseconds** (10 ms units).

(memory-map--assets)=
### Memory Map & Assets

- Static addresses used by this module include:
  - `RACER_PTR_ADDR`, `COURSE_ID_ADDR`, `OBJECTS_PTR_ADDR`, `CHECKPOINT_PTR_ADDR`, `CLOCK_DATA_PTR`, `CAMERA_PTR_ADDR`.
- **Courses**: `courses.json` maps course IDs → directory names. Per-course assets:
  - `course_collision.kcl` (KCL collision)
  - `course_map.nkm` (NKM map/logic)

(caching-model)=
### Caching Model

Two decorators reduce emulator I/O:

- `@frame_cache` — Cache **once per emulator tick** (`emu.get_ticks()`). Recompute on tick change.
- `@game_cache` — Cache for the **entire run** (process lifetime).

> **Caveat:** Both caches ignore argument values and memoize a **single** result. Pass stable arguments (e.g., a constant `device`) or create uncached variants for argument-sensitive results.

(device-handling)=
### Device Handling

Functions that return tensors accept a `device` argument. Use consistent devices across calls for best performance and cache coherence. KCL/NKM tensors are loaded to the device used at first call and are **game-cached**.

---

(constants)=
## Constants

| Name | Value | Description |
|------|-------|-------------|
| `SCREEN_WIDTH` | `256` | DS top-screen width in pixels |
| `SCREEN_HEIGHT` | `192` | DS top-screen height in pixels |
| `Z_FAR` | `1000.0` | Far-plane distance used in projection |
| `Z_NEAR` | `0.0` | Near-plane distance used in projection |
| `Z_SCALE` | `10.0` | Depth normalization scale factor |
| `RACER_PTR_ADDR` | `0x0217ACF8` | Memory address of racer pointer |
| `COURSE_ID_ADDR` | `0x23CDCD8` | Current course ID (byte) |
| `OBJECTS_PTR_ADDR` | `0x0217B588` | Object array metadata base |
| `CHECKPOINT_PTR_ADDR` | `0x021755FC` | Checkpoint manager pointer |
| `CLOCK_DATA_PTR` | `0x0217AA34` | Clock data base pointer |
| `CAMERA_PTR_ADDR` | `0x0217AA4C` | Camera pointer |
| `FLAG_DYNAMIC` | `0x1000` | Object flag: dynamic |
| `FLAG_MAPOBJ` | `0x2000` | Object flag: map object |
| `FLAG_ITEM` | `0x4000` | Object flag: item |
| `FLAG_RACER` | `0x8000` | Object flag: racer |

---

(api-reference)=
## API Reference

The API is grouped by responsibility. Where applicable, returns are `torch.Tensor` on the specified `device`.

(decorators)=
### Decorators

#### `frame_cache(func)`
Caches the wrapped function’s single result **per emulator tick** (`emu.get_ticks()`). Recomputes on tick change. Good for expensive reads that are stable within a frame.

#### `game_cache(func)`
Caches the wrapped function’s single result **for the entire run**. Use for course files and other static data.

#### `safe_object(func)`
For functions taking `(emu, id, ...)`, returns `None` if the object appears deleted (null position pointer). Prevents invalid memory access patterns.

---

(utility)=
### Utility

#### `z_clip_mask(x: torch.Tensor) -> torch.Tensor`
Boolean mask for rows whose camera-space `Z` is within `[-Z_FAR, -Z_NEAR]`. Use to cull points outside the depth range.

---

(clock--course)=
### Clock & Course

#### `read_clock_ptr(emu)` *(game-cached)*  
Base pointer to the game’s clock data struct.

#### `read_clock(emu)` *(frame-cached)*  
Current game time in **centiseconds** (10ms units).

#### `get_current_course_id(emu)`  
Read current course ID (byte).

#### `get_course_path(id: int)`  
Resolve a course ID to a directory path via `utils/courses.json`. Raises if missing.

#### `load_current_kcl(emu, device)` *(game-cached)*  
Parse and load the **KCL collision** file for the current course into a `KCLTensor` on `device`.

#### `load_current_nkm(emu, device)` *(game-cached)*  
Parse and load the **NKM** map file into an `NKMTensor` on `device`.

---

(player--objects)=
### Player & Objects

#### `read_racer_ptr(emu, addr=RACER_PTR_ADDR)`  
Returns the address of the racer struct.

#### `read_position(emu, device)` *(frame-cached)*  
Player world position `(3,)` as a tensor.

#### `read_direction(emu, device)` *(frame-cached)*  
Player forward direction `(3,)` as a tensor.

#### `read_objects_array_max_count(emu, addr=OBJECTS_PTR_ADDR)`  
Max number of objects in the global array.

#### `read_objects_array_ptr(emu, addr=OBJECTS_PTR_ADDR)`  
Pointer to the object pointer array.

#### `read_object_offset(emu, id)`  
Compute per-object metadata entry offset.

#### `read_object_ptr(emu, id)`  
Pointer to object struct (0 if null).

#### `read_object_flags(emu, id)`  
Unsigned short flags for the object (type/state bits).

#### `read_object_position_ptr(emu, id)`  
Pointer to object’s position struct (0 if deleted).

#### `read_object_is_ignored(emu, id)`  
Returns `True` if the object is null or has the ignored bit set.

#### `read_object_is_deleted(emu, id)`  
Returns `True` if the object’s position pointer is null.

#### `read_object_position(emu, id, device)` *(safe_object + frame-cached)*  
Object world position `(3,)` or `None` if deleted.

#### `read_map_object_type_id(emu, id)` *(safe_object + frame-cached)*  
Signed short map-object type ID, or `None`.

#### `read_map_object_is_coin_collected(emu, id)` *(safe_object + frame-cached)*  
`True` if coin is collected, else `False`; or `None`.

#### `read_racer_object_is_ghost(emu, id)` *(safe_object + frame-cached)*  
`True` if racer object is in ghost state; or `None`.

#### `read_objects(emu)` *(frame-cached)*  
Scan and group objects into categories: `map_objects`, `racer_objects`, `item_objects`, `dynamic_objects`. Returns `dict[str, list[int]]`.

---

(camera--projection)=
### Camera & Projection

#### `read_camera_ptr(emu, addr=CAMERA_PTR_ADDR)` *(frame-cached)*  
Address of the active camera struct.

#### `read_camera_fov(emu)` *(frame-cached)*  
Camera field-of-view in **radians** (from 16-bit fixed-point).

#### `read_camera_aspect(emu)` *(frame-cached)*  
Aspect ratio (width/height).

#### `read_camera_position(emu, device)` *(frame-cached)*  
Camera world position `(3,)` including elevation offset.

#### `read_camera_target_position(emu, device)`  
Camera look-at target `(3,)`.

#### `read_model_view(emu, device)` *(frame-cached)*  
4×4 model-view matrix (row-major), derived from camera position/target.

#### `project_to_screen(emu, points, device)`  
Convenience wrapper that reads current camera, then calls the internal `_project_to_screen`. Returns `(N, 4)` → `[x_px, y_px, clip_z, normalized_depth]` in a `256×192` viewport.

> See also: `z_clip_mask(x)` to cull points outside view-space Z range.

---

(checkpoints)=
### Checkpoints

#### `read_checkpoint_ptr(emu, addr=CHECKPOINT_PTR_ADDR)` *(game-cached)*  
Pointer to checkpoint manager/state.

#### `read_current_checkpoint(emu)` *(frame-cached)*  
Current checkpoint index (unsigned byte).

#### `read_current_key_checkpoint(emu)` *(frame-cached)*  
Current **key** checkpoint index (signed byte).

#### `read_ghost_checkpoint(emu)` *(frame-cached)*  
Ghost checkpoint index (signed byte).

#### `read_ghost_key_checkpoint(emu)` *(frame-cached)*  
Ghost key checkpoint index (signed byte).

#### `read_current_lap(emu)` *(frame-cached)*  
Current lap index (0-based, signed byte).

#### `read_next_checkpoint(emu, checkpoint_count)` *(frame-cached)*  
Compute next checkpoint index with wrap-around to 0.

#### `read_checkpoint_positions(emu, device)` *(game-cached)*  
Tensor of shape `(C, 2, 3)` listing endpoints `[p1, p2]` per checkpoint in **3D**. Uses KCL floors to lift 2D endpoints to 3D by nearest-neighbor elevation.

#### `read_next_checkpoint_position(emu, device)` *(frame-cached)*  
Endpoints `(2,3)` for the next checkpoint segment.

#### `read_current_checkpoint_position(emu, device)` *(frame-cached)*  
Endpoints `(2,3)` for the current checkpoint segment.

#### `read_facing_point_checkpoint(emu, direction, device)` *(frame-cached)*  
Intersection point of a ray (from player in `direction`) with the **next checkpoint line** in **XZ**; returns 3D point.

#### `read_forward_distance_checkpoint(emu, device)` *(frame-cached)*  
Forward distance to next checkpoint along player’s forward vector (scalar tensor).

#### `read_left_distance_checkpoint(emu, device)` *(frame-cached)*  
Leftward distance to the next checkpoint (scalar tensor).

#### `read_direction_to_checkpoint(emu, device)` *(frame-cached)*  
Steering angle `atan(forward / left)` toward the next checkpoint (radians).

---

(obstacles-walls--offroad)=
### Obstacles (Walls / Offroad)

#### `read_facing_point_obstacle(emu, position=None, direction=None, device=None)` *(frame-cached)*  
Samples a **cone of rays** around the provided (or player’s) direction and finds the **nearest** intersection with **wall/offroad** triangles from KCL. Returns a 3D point or `None` if no intersections.

#### `read_forward_distance_obstacle(emu, device)` *(frame-cached)*  
Forward distance to the nearest wall/offroad obstacle; returns `+inf` (tensor) when no hit.

#### `read_left_distance_obstacle(emu, device)` *(frame-cached)*  
Leftward distance to nearest obstacle; `+inf` when no hit.

#### `read_right_distance_obstacle(emu, device)` *(frame-cached)*  
Rightward distance to nearest obstacle; `+inf` when no hit.

#### `read_checkpoint_distance_altitude(emu, device)` *(frame-cached)*  
Triangle altitude given the player position and the two endpoints of the **next** checkpoint. Useful as a lateral proximity measure to the segment.

---

(internal-helpers)=
### Internal Helpers

The following helpers are considered internal (subject to change):

- `_compute_orthonormal_basis(forward, reference=None, device=None)` → `(3,3)` rows `[right, up, forward]`.
- `_compute_model_view(camera_pos, camera_target_pos, device)` → `(4,4)` model-view matrix.
- `_project_to_screen(world_points, model_view, fov, aspect, device=None)` → `(N,4)` screen coordinates.
- `_convert_2d_checkpoints(P, source, dim=0)` → lift 2D endpoints to 3D using nearest floor elevation.

---

(examples)=
## Examples

(project-the-player-to-screen-space)=
### Project the Player to Screen Space

```python
pos = read_position(emu, device)
screen_pt = project_to_screen(emu, pos.unsqueeze(0), device=device)
x_px, y_px = screen_pt[0, 0].item(), screen_pt[0, 1].item()
```

(distances-to-next-checkpoint)=
### Distances to Next Checkpoint

```python
d_fwd = read_forward_distance_checkpoint(emu, device)  # scalar tensor
d_left = read_left_distance_checkpoint(emu, device)    # scalar tensor
steer = read_direction_to_checkpoint(emu, device)      # radians
```

(nearest-obstacle-ahead)=
### Nearest Obstacle Ahead

```python
d_obs = read_forward_distance_obstacle(emu, device)
if torch.isfinite(d_obs):
    print(f"Obstacle ahead in {float(d_obs):.2f} units")
else:
    print("No obstacle detected ahead")
```

---

(edge-cases--error-handling)=
## Edge Cases & Error Handling

- **Deleted / Ignored Objects**: Functions decorated with `safe_object` return `None` for deleted objects. Check result before use.
- **No Geometry**: If no wall/offroad triangles exist or raycasts miss, obstacle distances return `+inf` (tensor).
- **Empty Projection Inputs**: `_project_to_screen` returns an empty tensor for empty inputs.
- **Caching Caveat**: Cached functions ignore argument values. If you need different behavior per-argument, create separate uncached functions or structure your calls so arguments are stable.

---

(performance-notes)=
## Performance Notes

- Caching eliminates redundant memory reads within a frame and across the run.
- Geometry routines (ray casting, projection, distances) are vectorized in Torch. Prefer GPU/MPS devices when available.
- Keep devices consistent where cached data is shared (e.g., KCL/NKM loaded once with `@game_cache`).

---

(faq)=
## FAQ

**Q: Why are screen Y coordinates flipped?**  
A: DS screen uses a top-left origin; we map NDC to pixel space with `(1 - ndc_y)` to respect that convention.

**Q: Do I need to pass `device` everywhere?**  
A: Only to functions returning tensors. Consistency is important because some results are cached and tied to the first-used device.

**Q: I changed args but got a cached result — bug?**  
A: By design, `@frame_cache` and `@game_cache` memoize a single value and **ignore** argument values. Keep args stable or avoid caching for arg-sensitive functions.

**Q: What units are returned from memory reads?**  
A: FX32 (and similar) are converted to floats by helper functions before tensors are constructed.

---

(license)=
## License

This documentation is provided as part of the project’s repository. See the repository’s root `LICENSE` file for terms.
