# `mkds` — Mario Kart DS NKM & KCL Readers

> **Installable parser library for Mario Kart DS course files.**  
> Read *NKM* (course map) and *KCL* (collision) files with clean Python APIs. Designed for tooling, editors, analytics, and research.

- **PyPI:** `pip install mkds`
- **Modules:** `mkds.nkm`, `mkds.kcl`
- **Status:** Focused on correctness and readability. Torch extensions live outside this package (see [Torch Extensions (Optional)](#torch-extensions-optional)).

---

## Table of Contents

- [overview](#overview1)
- [Installation](#installation1)
- [Getting Started](#getting-started1)
- [Design \& Conventions](#design--conventions1)
  - [Binary Layouts](#binary-layouts)
  - [Fixed-Point Formats](#fixed-point-formats)
  - [Section Iteration Model](#section-iteration-model)
- [NKM — Course Map](#nkm--course-map)
  - [Quickstart](#quickstart)
  - [Header \& Section Offsets](#header--section-offsets)
  - [Sections API](#sections-api)
    - [Section (base)](#section-base)
    - [OBJI — Object Instances](#obji--object-instances)
    - [PATH — Path Metadata](#path--path-metadata)
    - [POIT — Path Points](#poit--path-points)
    - [STAG — Stage Info](#stag--stage-info)
    - [KTPS — Kart/Start Positions](#ktps--kartstart-positions)
    - [KTPJ — Respawn Positions](#ktpj--respawn-positions)
    - [KTP2 — Lap Checkpoint Points](#ktp2--lap-checkpoint-points)
    - [KTPC — Cannon/Pipe Destinations](#ktpc--cannonpipe-destinations)
    - [KTPM — Mission Points](#ktpm--mission-points)
    - [CPOI — Checkpoints](#cpoi--checkpoints)
    - [CPAT — Checkpoint Groups](#cpat--checkpoint-groups)
    - [IPOI — Item Points](#ipoi--item-points)
    - [IPAT — Item Groups](#ipat--item-groups)
    - [EPOI — CPU Path Points](#epoi--cpu-path-points)
    - [EPAT — CPU Grouping](#epat--cpu-grouping)
    - [MEPO — Minigame Enemy Points](#mepo--minigame-enemy-points)
    - [MEPA — Minigame Grouping](#mepa--minigame-grouping)
    - [AREA — Areas / Zones](#area--areas--zones)
    - [CAME — Cameras](#came--cameras)
    - [NKM — Top-Level Parser](#nkm--top-level-parser)
- [KCL — Collision](#kcl--collision)
  - [Quickstart](#quickstart-1)
  - [Header Layout](#header-layout)
  - [API](#api)
    - [`PrismsBase`](#prismsbase)
    - [`KCLBase`](#kclbase)
    - [`Prisms`](#prisms)
    - [`KCL`](#kcl)
- [Torch Extensions (Optional)](#torch-extensions-optional)
- [Edge Cases \& Version Notes](#edge-cases--version-notes)
- [FAQ](#faq)
- [License](#license)

---
(overview1)=
## Overview

The `mkds` package provides binary readers for **Mario Kart DS** course files:

- **NKM** — Course map logic and gameplay data (objects, paths, checkpoints, cameras, etc.).
- **KCL** — Collision meshes using triangular prisms with an octree index.

This library focuses on **structured access** to course data with minimal surprises:
- Sections are parsed into **Python objects** with typed lists/fields.
- Iteration is predictable via a base `Section` class for fixed-stride sections.
- Field readers (`read_u16`, `read_fx32`, etc.) hide fixed-point details.

---
(installation1)=
## Installation1

```bash
pip install mkds
```

> Python ≥3.10 is recommended.

---
(getting-started1)=
## Getting Started

```python
from mkds.nkm import NKM
from mkds.kcl import KCL

# NKM
nkm = NKM.from_file("my_course.nkm")
print(len(nkm._OBJI))           # number of object instances
print(nkm._STAG.amt_of_laps)    # lap count

# KCL
kcl = KCL.from_file("course_collision.kcl")
print(len(kcl.prisms))          # number of prisms
print(kcl.positions[0])         # first vertex position
```

---
(design--conventions1)=
## Design & Conventions
(binary-layouts)=
### Binary Layouts

- **NKM** contains a header with **relative offsets** to each section. `NKM` adds the fixed header length (0x4C) to compute absolute file positions and slices the byte array to create each section object.
- **KCL** contains offsets to positions, normals, prisms, and the octree block region. `KCLBase` parses positions/normals and exposes prism attributes as Python lists.
(fixed-point-formats)=
### Fixed-Point Formats

- Many numeric fields are fixed-point: **Fx16** and **Fx32**. Readers in `mkds.utils` convert to Python floats transparently.
- Vector readers (e.g., `read_vector_3d_fx32`) return `(x, y, z)` tuples in floating-point.
(section-iteration-model)=
### Section Iteration Model

- Most NKM sections (except **STAG**) implement:
  - An 8-byte header (`magic`, `entry_count`).
  - Fixed-size entries (a **stride**).  
- The `Section` base class:
  - Parses `entry_count` from the header.
  - Provides `__len__` and `__iter__` to iterate over raw entry bytes.
  - Concrete subclasses decode per-entry fields into Python lists.

---

(nkm--course-map)=
## NKM — Course Map

(quickstart)=
### Quickstart

```python
from mkds.nkm import NKM

nkm = NKM.from_file("my_course.nkm")

# Example: iterate object IDs
for oid in nkm._OBJI.object_id:
    print("Object ID:", oid)

# Example: checkpoint segment endpoints (2D vectors)
p1 = nkm._CPOI.position1[0]
p2 = nkm._CPOI.position2[0]
```

(header--section-offsets)=
### Header & Section Offsets

- `NKM._header_offset = 0x4C` (header length).  
- Section offsets in the header (at 0x08..0x48) are **relative to the end of the header**; the parser **adds 0x4C** to get absolute positions.
- The canonical order in this implementation:
  `OBJI, PATH, POIT, STAG, KTPS, KTPJ, KTP2, KTPC, KTPM, CPOI, CPAT, IPOI, IPAT, EPOI, EPAT, AREA, CAME`.

(sections-api)=
### Sections API

Below, fields are Python lists with per-entry values unless noted.

(section-base)=
#### Section (base)

- **Purpose:** Common scaffolding for fixed-stride NKM sections.  
- **Attributes:**
  - `data`: Raw section bytes (header + entries)
  - `stride`: Entry size in bytes
  - `entry_count`: Count parsed from header (UInt32 at offset 0x04)
- **Iteration:** Iterates *entry slices* (`data[8 + i*stride : 8 + (i+1)*stride]`).

---

(obji--object-instances)=
#### OBJI — Object Instances

- **Stride:** `0x3C` bytes
- **Purpose:** Placement of map objects (decorations, interactives, item boxes, etc.).
- **Fields:**
  - `rot_vec1: list[tuple[float,float,float]]` — position (X,Y,Z) @0x00 (VecFx32)
  - `rot_vec2: list[tuple[float,float,float]]` — rotation vector @0x0C
  - `scale_vec: list[tuple[float,float,float]]` — scale vector @0x18
  - `object_id: list[int]` — @0x24
  - `route_id: list[int]` — @0x26 (0xFFFF ⇒ no route)
  - `object_settings: list[list[int]]` — four UInt32s @0x28..0x37
  - `show_in_time_trials: list[int]` — UInt32 @0x38
- **Notes:** Settings meaning depends on `object_id`. Routes link to `PATH`/`POIT`.

---

(path--path-metadata)=
#### PATH — Path Metadata

- **Stride:** `0x04` bytes
- **Purpose:** Route descriptors pointing into `POIT` point streams.
- **Fields:**
  - `route_id: list[int]` — Byte @0x00
  - `has_loop: list[bool]` — Byte @0x01 (**implementation uses `!= 1`**; canonical spec is `== 1` for loop)
  - `point_count: list[int]` — UInt16 @0x02

---

(poit--path-points)=
#### POIT — Path Points

- **Stride:** `0x14` bytes
- **Purpose:** 3D points forming the routes.
- **Fields:**
  - `position: list[tuple[float,float,float]]` — VecFx32 @0x00
  - `point_index: list[int]` — Byte @0x0C
  - `unknown1: list[int]` — Byte @0x0D
  - `point_duration: list[int]` — Int16 @0x0E
  - `unknown2: list[int]` — UInt32 @0x10

---

(stag--stage-info)=
#### STAG — Stage Info

- **Unique:** *No section header*. Fixed-size 0x2C struct.
- **Fields (high level):**
  - `track_id: int`, `amt_of_laps: int`, `fog_enabled: bool`, fog params, KCL colors (placeholders), etc.
- **Notes:** Color fields (GXRgb) are left as `None` in this implementation.

---

(ktps--kartstart-positions)=
#### KTPS — Kart/Start Positions

- **Stride:** `0x1C` bytes
- **Fields:** `position`, `rot_vec`, `padding`, `start_position_index`.

---


(ktpj--respawn-positions)=
#### KTPJ — Respawn Positions

- **Stride:** `0x20` bytes
- **Fields:** `position`, `rot_vec`, `enemy_position_id` (EPOI), `item_position_id` (IPOI), `respawn_id` (may be absent in very old versions).

---

(ktp2--lap-checkpoint-points)=
#### KTP2 — Lap Checkpoint Points

- **Stride:** `0x1C` bytes
- **Fields:** `position`, `rot_vec`, `padding`, `index` (often 0xFFFF).

---

(ktpc--cannonpipe-destinations)=
#### KTPC — Cannon/Pipe Destinations

- **Stride:** `0x1C` bytes
- **Fields:** `position`, `rot_vec`, `unknown`, `cannon_index`.

---

(ktpm--mission-points)=
#### KTPM — Mission Points

- **Stride:** `0x1C` bytes
- **Fields:** `position`, `rot_vec`, `padding`, `index`.

---

(cpoi--checkpoints)=
#### CPOI — Checkpoints

- **Stride:** `0x24` bytes
- **Fields:**
  - `position1`, `position2`: 2D vectors (VecFx32) @0x00, 0x08
  - `sinus`, `cosinus`, `distance`: Fx32 @0x10..0x18
  - `section_data1`, `section_data2`: UInt16 @0x1C..0x1E
  - `key_id`: UInt16 @0x20 (0=lap, 0xFFFF=none, otherwise keyed checkpoint)
  - `respawn_id`: Byte @0x22
  - `unknown`: Byte @0x23

---

(cpat--checkpoint-groups)=
#### CPAT — Checkpoint Groups

- **Stride:** `0x0C` bytes
- **Fields:** `point_start`, `point_length`, `next_group[3]`, `prev_group[3]`, `section_order`.

---

(ipoi--item-points)=
#### IPOI — Item Points

- **Stride:** `0x14` bytes
- **Fields:** `position`, `point_scale` (Fx32), `unknown` (UInt32).

---

(ipat--item-groups)=
#### IPAT — Item Groups

- **Stride:** `0x0C` bytes
- **Fields:** `point_start`, `point_length`, `next_group[3]`, `prev_group[3]`, `section_order`.

---

(epoi--cpu-path-points)=
#### EPOI — CPU Path Points

- **Stride:** `0x18` bytes
- **Fields:** `position`, `point_scale` (Fx32), `drifting` (Int16), `unknown1` (UInt16), `unknown2` (UInt32).

---

(epat--cpu-grouping)=
#### EPAT — CPU Grouping

- **Stride:** `0x0C` bytes
- **Fields:** `point_start`, `point_length`, `next_group[3]`, `prev_group[3]`, `section_order`.

---

(mepo--minigame-enemy-points)=
#### `MEPO` — Minigame Enemy Points

- **Stride:** `0x18` bytes
- **Fields:** `position`, `point_scale`, `drifting` (Int32), `unknown` (UInt32).

---

(mepa--minigame-grouping)=
#### MEPA — Minigame Grouping

- **Stride:** `0x14` bytes
- **Fields:** `point_start`, `point_length`, `next_group[8]`, `prev_group[8]`.

---

(area--areas--zones)=
#### AREA — Areas / Zones

- **Stride:** `0x48` bytes
- **Fields:** `position` (center), `length_vec`, `x_vec`, `y_vec`, `z_vec`, camera linkage, and several unknowns. `area_type` is left as `None` placeholder.

---

(came--cameras)=
#### CAME — Cameras

- **Stride:** `0x4C` bytes
- **Fields:** 3D positions, rotation, FOV begin/end (+ sine/cosine), zoom, type, linked route, speeds, duration, next camera, intro-pan indicator, etc.
- **Notes:** Several precomputed fields (sin/cos) are preserved as-is.

---

(nkm--top-level-parser)=
#### NKM — Top-Level Parser

```python
from mkds.nkm import NKM

nkm = NKM.from_file("my_course.nkm")
# Access parsed sections:
nkm._OBJI, nkm._PATH, nkm._POIT, nkm._STAG, nkm._KTPS, nkm._KTPJ, nkm._KTP2, \
nkm._KTPC, nkm._KTPM, nkm._CPOI, nkm._CPAT, nkm._IPOI, nkm._IPAT, nkm._EPOI, \
nkm._EPAT, nkm._AREA, nkm._CAME
```

- **Construction:** Slices raw file by computed offsets to build section objects.
- **Classmethod:** `NKM.from_file(path)` opens and parses bytes in one call.

---

(kcl--collision)=
## KCL — Collision

(quickstart-1)=
### Quickstart

```python
from mkds.kcl import KCL

kcl = KCL.from_file("course_collision.kcl")
print("Prisms:", len(kcl.prisms))
print("First prism vertex index:", kcl.prisms.pos_i[0])
```

(header-layout)=
### Header Layout

| Offset | Type | Name | Description |
|---:|:---:|:---|:---|
| 0x00 | u32 | positions_offset | Start of position array |
| 0x04 | u32 | normals_offset | Start of normal array |
| 0x08 | u32 | prisms_offset | Start of prism array |
| 0x0C | u32 | block_data_offset | Start of octree blocks |
| 0x10 | f32 | prism_thickness | Depth of triangular prism along normal |
| 0x14 | Vec3 | area_min_pos | Min corner of model bounding box |
| 0x20 | u32 | area_x_width_mask | X-axis mask for octree |
| 0x24 | u32 | area_y_width_mask | Y-axis mask for octree |
| 0x28 | u32 | area_z_width_mask | Z-axis mask for octree |
| 0x2C | u32 | block_width_shift | Leaf block size (shift) |
| 0x30 | u32 | area_x_blocks_shift | Root child index shift (Y) |
| 0x34 | u32 | area_xy_blocks_shift | Root child index shift (Z) |
| 0x38 | f32? | sphere_radius | Optional, not parsed here |

(api)=
### API

(prismsbase)=
#### PrismsBase

Represents the raw prism table. Each entry (stride **0x10**) has:
- `height: list[float]`
- `pos_i: list[int]`
- `fnrm_i, enrm1_i, enrm2_i, enrm3_i: list[int]` — normal indices
- `attributes: list[list[int]]` — parsed bitfields via `parse_attributes(bits)`:
  - `[ map_2d_shadow, light_id(1-3), ignore_drivers, collision_variant, collision_type, ignore_items, is_wall, is_floor ]`

Constructors:
- `PrismsBase.from_bytes(data: bytes) -> PrismsBase` — parse a contiguous prism region.
- `PrismsBase.parse_attributes(bits: int) -> list[int]` — helper for attribute bit slicing.

(kclbase)=
#### KCLBase

Top-level parser for collision files (bytes-level). Responsibilities:
- Parse header offsets and metadata.
- Parse **positions** (VecFx32) and **normals** (VecFx16) based on max indices found in prisms.
- Build a `prisms` instance (class attribute `prism_cls` controls which concrete class is used).

Helpers:
- `_parse_positions(data, prisms, positions_offset)` → `list[tuple[float,float,float]]`
- `_parse_normals(data, prisms, normals_offset)` → `list[tuple[float,float,float]]`
- `search_block(point)` → *leaf block offset* within `block_data` containing `point`, or `None` if outside masks.

> **Leaf encoding note:** In octree leaves, prism indices are stored as **1-based** UInt16 values terminated by `0x0000`. When reading, subtract 1; stop at -1 sentinel.

(prisms)=
#### Prisms

A convenience subclass of `PrismsBase` that adds:
- `__len__()`, `__iter__()`
- `__getitem__(idx)` → returns all attributes for prism `idx`

(kcl)=
#### KCL

A convenience subclass of `KCLBase` that:
- Uses `Prisms` as `prism_cls`.
- Adds a human-readable `__str__` summary.
- Provides `from_file(path)` to open+parse in one call.

---

(torch-extensions-optional)=
## Torch Extensions (Optional)

This package focuses on **I/O**. If you need tensor-native processing (GPU/MPS), consider separate extensions (not part of `mkds`):

- **`PrismsTensor`** — wraps prism arrays as `torch.Tensor` with convenience boolean masks (`is_wall`, `is_floor`, etc.).
- **`KCLTensor`** — adds triangle reconstruction and batched queries (e.g., nearest triangles, point→triangle distances).
- **`CPOITensor` / `NKMTensor`** — expose NKM checkpoint positions as tensors for geometric operations.

> Example import paths may vary in your codebase (e.g., `utils.kcl_torch` / `utils.nkm_torch`). These are **not** included in the `mkds` PyPI package.

---

(edge-cases--version-notes)=
## Edge Cases & Version Notes

- **NKM `PATH.has_loop`:** The implementation reads `has_loop = (byte != 1)` to preserve historical behavior. Canonical spec treats `1` as *loop enabled*. Harmonize across your tools if needed.
- **NKM `STAG`:** Uses placeholders (`None`) for colors. If you need colors, implement GXRgb decoding and assign real values.
- **Old Beta Tracks:** Some fields (e.g., `KTPJ.respawn_id`) may not exist; handle absent bytes when writing broad-compat parsers.
- **KCL Leaf Blocks:** Outside the octree masks (`area_*_width_mask`) return `None` in `search_block`.
- **Fixed-Point Precision:** Fx16/Fx32 values are converted to floats, which is adequate for tooling; games may use integer arithmetic internally.

---

(faq)=
## FAQ

**Q: How do I list all cameras that follow a route?**  
A: In `nkm._CAME`, filter entries where `linked_route[i] != 0xFFFF`.

**Q: How do I determine if a prism is a wall or floor?**  
A: Use `prisms.attributes[i][6]` and `[7]` respectively (or named tensor properties in your Torch extension).

**Q: Can I rebuild triangles directly from KCL?**  
A: Yes, but `mkds` itself doesn’t compute them. Use a tensor extension (see *Torch Extensions*) or reconstruct from `positions`, `normals`, and prism `height`/normals like in the example code in your project.

**Q: Are indices 0-based?**  
A: Prism indices in leaves are **1-based** and terminated by 0; everywhere else indices into arrays are 0-based in the parsed lists.

---

(license)=
## License

This documentation accompanies the `mkds` parsing library. See the repository’s `LICENSE` file for terms.
