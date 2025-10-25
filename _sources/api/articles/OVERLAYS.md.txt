(mkds-overlay-system-extensible-overlays-for-mario-kart-ds)=
# MKDS Overlay System — Extensible Overlays for Mario Kart DS

> **Composable overlay framework for the DeSmuME renderer.**  
> Compute geometry from live emulator memory, project to screen space, and enqueue drawing primitives for a GTK/Cairo render loop. Easily add your own overlays by writing a single `overlay(emu, device)` function.

- **Focus:** Overlays (`utils/overlay.py`) and drawing stack (`utils/draw.py`)
- **Renderer example:** `main.py` (GTK window & render loop integration)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [Execution Flow](#execution-flow)
  - [Threading Model](#threading-model)
- [Getting Started](#getting-started)
- [Coordinate System & Screen Space](#coordinate-system--screen-space)
- [Overlay API (Built-ins)](#overlay-api-built-ins)
  - [`collision_overlay`](#collision_overlay)
  - [`raycasting_overlay`](#raycasting_overlay)
  - [`camera_overlay`](#camera_overlay)
  - [`checkpoint_overlay_1`](#checkpoint_overlay_1)
  - [`checkpoint_overlay_2`](#checkpoint_overlay_2)
  - [`player_overlay`](#player_overlay)
  - [`stats_overlay`](#stats_overlay)
- [Drawing Primitives & Queue](#drawing-primitives--queue)
  - [`draw_points`](#draw_points)
  - [`draw_lines`](#draw_lines)
  - [`draw_triangles`](#draw_triangles)
  - [`draw_text`](#draw_text)
  - [`draw_paragraph`](#draw_paragraph)
  - [`consume_draw_stack`](#consume_draw_stack)
  - [Decorator: `draw_stack_op`](#decorator-draw_stack_op)
- [Creating Your Own Overlay](#creating-your-own-overlay)
- [Integration Example (GTK)](#integration-example-gtk)
- [Performance Notes](#performance-notes)
- [FAQ](#faq)
- [License](#license)

---

(overview)=
## Overview

The overlay system is designed to be **highly extensible**: write small, pure functions that read live game state from the emulator, project points to **screen space (256×192)**, and enqueue Cairo drawing operations. The GTK renderer consumes those operations on each `draw` event, layering them over the emulator frame.

Core pieces:

- `utils/overlay.py`: A set of example overlays showing best practices.
- `utils/draw.py`: A composable **draw-queue** built on a Cairo `Context` and a decorator to enqueue draw calls from any thread.
- `main.py`: A concrete GTK + DeSmuME render loop showing how to plug in overlays.

---

(how-it-works)=
## How It Works

(execution-flow)=
### Execution Flow

1. **Emulator tick**: `emu.cycle()` advances the game by one frame.
2. **Overlay evaluation** (worker thread): Each overlay (`overlay(emu, device)`) reads memory, computes geometry, projects to screen, and **enqueues** draw ops via `draw_*` helpers.
3. **GTK draw event** (UI thread): `on_draw_main` calls `consume_draw_stack(ctx)`, which dequeues closures and draws them onto a transparent overlay surface. That surface is composited over the emulator screen.

(threading-model)=
### Threading Model

- Overlays typically run in a **worker thread** (see `worker()` and the `tick()` in `main.py`).  
- `draw_queue` is a thread-safe `queue.Queue`; decorated `draw_*` functions push **closures** that will be executed later in the GTK thread with a Cairo `Context`.
- The GTK `draw` callback maintains an **overlay surface cache** to reuse the last overlay frame when no new draw ops were enqueued (reduces flicker and work).

---

(getting-started)=
## Getting Started

```python
from utils.overlay import (
    collision_overlay,
    checkpoint_overlay_1, checkpoint_overlay_2,
    player_overlay, raycasting_overlay, camera_overlay,
    stats_overlay,
)

# In your run loop, pass a list of overlays:
run_emulator(
    generate_trainer,
    [
        collision_overlay,
        checkpoint_overlay_1,
        checkpoint_overlay_2,
        # player_overlay,
        # raycasting_overlay,
        # camera_overlay,
        # stats_overlay,   # example: queue text/numbers
    ],
)
```

> Each overlay is called like `overlay(emu, device)` and should **only enqueue** draw ops. It must not block.

---

(coordinate-system--screen-space)=
## Coordinate System & Screen Space

- **World space** follows MKDS conventions (Y-up).
- **Screen space** is **256×192** pixels; origin at **top-left** (X→right, Y→down).
- Use `project_to_screen(emu, points, device)` to map `(N,3)` world coordinates → `(N,4)` screen data `[x_px, y_px, clip_z, depth_norm]`.
- Use `z_clip_mask(screen_tensor)` to cull points outside near/far planes: `(clip_z < -Z_NEAR) & (clip_z > -Z_FAR)`.
- Most overlays export the **third channel** in `draw_points`/`draw_lines` buffers as a *scale/depth* dimension used by the primitive (see below).

---

(overlay-api-built-ins)=
## Overlay API (Built-ins)

> All overlays follow the signature: `overlay(emu: DeSmuME, device=None) -> None`.

(collision_overlay)=
### `collision_overlay`

**Purpose:** Draw edges of collision triangles around the player, colored by attribute (walls and offroad types).

**Flow:**
1. Load KCL (`load_current_kcl`) and player `position`.
2. `indices = kcl.search_triangles(position)` finds nearby triangle indices (octree leaf).
3. Filter by attributes:
   - `is_wall` → magenta `(1, 0, 1)`
   - `collision_type in {2,3,5}` (offroad) → pink `(1, 0, 0.3)`
4. Project triangle vertices `v1, v2, v3` to screen; **z-clip**, then emit edges via `draw_triangles`.

**Notes:**
- Third component passed to draw is the **normalized depth** (used for stylized scaling).

---

(raycasting_overlay)=
### `raycasting_overlay`

**Purpose:** Demonstrate dynamic raycasting to the nearest obstacle in front of the kart, smoothing the intersection point over time.

**Flow:**
1. Read player `position`.
2. `read_facing_point_obstacle` samples a **cone** of rays to find the nearest hit (walls/offroad).
3. Smooth a global `current_point` toward the latest hit via `interpolate` (0.1 factor).
4. Compute forward/left/right distances for logging or additional overlays.

**Notes:**
- This overlay **does not draw** by itself; it prepares values for other overlays or logs. Add your own `draw_points` if desired.

---

(camera_overlay)=
### `camera_overlay`

**Purpose:** Visualize the current **camera target** position.

**Flow:**
1. Read `read_camera_target_position`.
2. Project to screen and plot a red point via `draw_points` (larger radius).

---

(checkpoint_overlay_1)=
### `checkpoint_overlay_1`

**Purpose:** Draw a **line segment** between the two endpoints of the **next checkpoint**, aligned to the player’s current Y (so it’s visible in 2D screen space).

**Flow:**
1. `read_next_checkpoint_position` → `(2,3)` endpoints.
2. Replace endpoints’ Y with player Y (visual flattening).
3. Project to screen, **z-clip**; if one endpoint remains, draw a single green dot.
4. Otherwise, concat `[x, y, depth_norm, clip_z]` and emit via `draw_lines` (green).

---

(checkpoint_overlay_2)=
### `checkpoint_overlay_2`

**Purpose:** Draw a **ray** from the kart to its intersection with the next checkpoint boundary.

**Flow:**
1. `read_direction` and `read_facing_point_checkpoint` → intersection point.
2. Project intersection to screen, z-clip, and draw a green point.
3. Project current player position, normalize both to a small Z for visibility, and draw a **blue line** from kart → intersection.

---

(player_overlay)=
### `player_overlay`

**Purpose:** Scatter-plot visible objects by category: `map_objects`, `racer_objects`, `item_objects`, `dynamic_objects`.

**Flow:**
1. `read_objects()` groups object IDs by flags.
2. For each group, collect positions with `read_object_position` (skips deleted/ignored via `safe_object`).
3. Project to screen, z-clip, and draw points in group color.

**Notes:**
- Uses four palette colors; colors broadcast to all points per group.

---

(stats_overlay)=
### `stats_overlay`

**Purpose:** Example overlay that reads stats **without drawing** (clock, forward distances). In the GTK example, textual stats are drawn in the `on_draw_main` callback using `draw_paragraph` directly (outside the queue).

**Suggested:** Convert to overlay by formatting stats and calling `draw_paragraph` to enqueue text if you prefer everything in the queue.

---

(drawing-primitives--queue)=
## Drawing Primitives & Queue

All drawing helpers live in `utils/draw.py`. They do **not** draw immediately. Instead, they’re decorated with `@draw_stack_op`, so calling them enqueues a closure that will be executed with a Cairo `Context` in the GTK draw thread.

(draw_points)=
### `draw_points`

```python
draw_points(pts: np.ndarray, colors: np.ndarray, radius_scale: float | np.ndarray)
```

- **Inputs:**
  - `pts`: shape `(N, 3)` or `(3,)` → `[x_px, y_px, z]`
    - **`z`** acts as a scale factor for the dot radius: `radius = radius_scale * z`
  - `colors`: shape `(N, 3)` or `(3,)` in **[0,1] RGB**; single color is broadcast to all points.
  - `radius_scale`: typically a **float**. (Per-point arrays are not currently indexed per row.)
- **Behavior:** Fills a disk at each point.

(draw_lines)=
### `draw_lines`

```python
draw_lines(pts1: np.ndarray, pts2: np.ndarray, colors: np.ndarray, stroke_width_scale=1.0)
```

- **Inputs:**
  - `pts1`, `pts2`: shapes `(N,3)` or `(3,)`. Only **X,Y** are used for line endpoints.
  - `colors`: `(N,3)` or `(3,)` RGB; single color broadcasts.
  - `stroke_width_scale`: Cairo line width.
- **Behavior:** Draws segments `pts1[i] → pts2[i]` per batch entry.

(draw_triangles)=
### `draw_triangles`

```python
draw_triangles(pts1: np.ndarray, pts2: np.ndarray, pts3: np.ndarray, colors: np.ndarray)
```

- **Inputs:** Three point arrays, shape `(N,3)` each. `colors` is `(3,)` or `(N,3)` RGB.
- **Behavior:** Renders triangle **edges** by internally enqueuing three `draw_lines` calls (one per edge).

(draw_text)=
### `draw_text`

Immediate text drawing helper (not queued).

```python
draw_text(text, pos=(x,y), color=(r,g,b), alpha=1.0, font_size=12, font_family="Sans")
```

- Use directly in a Cairo context (e.g., from `on_draw_main`) for HUD-like text that you want to render **every frame**.

(draw_paragraph)=
### `draw_paragraph`

Immediate block text (multiple lines).

```python
draw_paragraph(text, pos=(x,y), color, alpha, font_size, vertical_spacing, font_family)
```

(consume_draw_stack)=
### `consume_draw_stack`

Consume and execute up to `max_items` enqueued draw operations.

```python
num = consume_draw_stack(ctx: cairo.Context, max_items: int | None = None) -> int
```

- Returns the number of ops executed.
- Use from the GTK draw callback to render overlays on a transparent surface.

(decorator-draw_stack_op)=
### Decorator: `draw_stack_op`

```python
@draw_stack_op
def draw_points(ctx: Context, ...):
    ...
```

- Wraps a function of shape `(ctx, *args)` into an **enqueuer** that accepts `*args` **now** and pushes a closure to call the original with a Cairo `Context` **later**.
- This decouples CPU-side geometry work from UI-thread rendering.

---

(creating-your-own-overlay)=
## Creating Your Own Overlay

An overlay is a simple function:

```python
import numpy as np
import torch
from utils.draw import draw_points
from utils.memory import read_position, project_to_screen, z_clip_mask

def my_overlay(emu, device=None):
    # 1) Read from emulator
    pos = read_position(emu, device=device).unsqueeze(0)  # (1,3)

    # 2) Project to screen
    sp = project_to_screen(emu, pos, device=device)       # (1,4) [x,y,clip_z,depth_norm]

    # 3) Depth cull
    mask = z_clip_mask(sp)
    sp = sp[mask]
    if sp.shape[0] == 0:
        return

    # 4) Prepare np arrays (x, y, depth_norm) and color
    pts = torch.cat([sp[:, :2], sp[:, 3, None]], dim=-1).cpu().numpy()
    color = np.array([0.2, 0.8, 0.6])

    # 5) Enqueue draw op
    draw_points(pts, colors=color, radius_scale=6.0)
```

**Tips:**
- **Never block** in overlays. Do reads/computation; enqueue and return.
- **Convert to NumPy** right before enqueuing (`.detach().cpu().numpy()`).
- **Use `z_clip_mask`** after projection to avoid drawing behind-camera artifacts.
- To draw edges, prepare two `(N,3)` arrays `(pts1, pts2)` and call `draw_lines`.

---

(integration-example-gtk)=
## Integration Example (GTK)

`main.py` shows a complete integration with GTK and DeSmuME:

- Creates an `EmulatorWindow` with a `Gtk.DrawingArea` scaled by `SCALE`.
- On each `draw`:
  - Renders the emulator frame into the Cairo target.
  - Creates a **transparent overlay surface**, clears it, then calls `consume_draw_stack(overlay_ctx)`.
  - Caches the overlay surface when **no draws** occurred (saves work).
  - Composites the overlay over the emulator frame via `cairo.OPERATOR_OVER`.
  - Optionally draws HUD text via `draw_paragraph` (immediate helper).
- A timed callback (`GLib.timeout_add(16, tick)`) advances the emulator at ~60 FPS, pushes the current `emu` into the worker queue, and schedules a redraw.
- The worker thread pulls `emu` and runs each overlay in order, enqueuing draw ops via the queue-safe decorators.

**Key bindings:** Mapped to DS keys via `pynput` and `desmume.controls` (see `KEY_MAP`).

---

(performance-notes)=
## Performance Notes

- **Minimize device↔host transfers**: keep tensors on device during math; transfer to CPU only when creating NumPy arrays for drawing.
- **Cull early**: use `z_clip_mask` prior to NumPy conversion.
- **Batch operations**: project arrays of points at once; avoid per-point projection.
- **Overlay surface cache**: already implemented — reuses last overlay frame when nothing changed.
- **No anti-aliasing**: overlay uses `cairo.ANTIALIAS_NONE` and `FILTER_NEAREST` to preserve the DS aesthetic.

---

(faq)=
## FAQ

**Q: Do I need to call `draw_*` with a Cairo context?**  
A: No. The decorator enqueues a closure; the GTK thread supplies the `Context` later.

**Q: Why does `draw_points` use the 3rd channel of `pts`?**  
A: It scales the point radius (`radius_scale * z`) to hint depth/parallax. For fixed-size markers, pass a constant `z=1` per point.

**Q: Can I pass per-point radii?**  
A: The current implementation expects a **float** `radius_scale`; per-point scaling isn’t indexed per row. If you need it, extend the function to accept a per-point array and index it inside the loop.

**Q: Where do `project_to_screen`, `z_clip_mask`, etc. come from?**  
A: They’re provided by the memory/geometry utilities (`utils.memory`), which wrap camera pose reads and projection math.

**Q: What if my overlay needs previous-frame state?**  
A: Use module-level globals (as `raycasting_overlay` does with `current_point`) or keep a small state object; be mindful of the worker thread context.

---

(license)=
## License

This overlay system documentation is part of the project repository. See `LICENSE` in the repo root for terms.
