"""Parallel trainer for Mario Kart DS agents using DeSmuME, multiprocessing,
shared memory frame streaming, and optional live GTK visualization.

This module orchestrates end-to-end evaluation and evolution of a population of
neural network controllers (NEAT-style) for Mario Kart DS. It supports three
execution modes per evaluated individual:

  1) **Headless** (`run_process`) — fast evaluation with no display.
  2) **Display worker** (`run_window_process`) — renders frames and writes them
     into a per-process shared-memory buffer; no GTK loop.
  3) **Display host** (`run_window_host_process`) — renders frames, writes them
     into shared memory, and owns the GTK window that tiles and presents all
     *display-enabled* workers in real time.

Key concepts
------------
- **Shared memory frames:** Each display-enabled process writes an RGBX
  framebuffer of shape ``(SCREEN_HEIGHT, SCREEN_WIDTH, 4)`` (dtype
  ``np.uint8``) to a POSIX shared-memory segment named ``f"emu_frame_{id}"``.
  The host window process opens these buffers read-only for display tiling.
- **Overlays:** Optional per-frame overlays are computed off the main emulation
  loop by a single background thread fed via a queue. Overlays are composited in
  the worker before writing to shared memory using
  :func:`src.visualization.window.on_draw_memoryview`.
- **Statistics / fitness:** Each process records split times and distances at
  track checkpoints as a ``dict[int, list[tuple[float, float]]]`` mapping
  ``checkpoint_id -> [(delta_time, distance_at_split), ...]``. A simple fitness
  function sums the recorded distances.
- **Batching & evolution:** :func:`run_training_session` evaluates a subset
  (batch) of the population in parallel (bounded by ``num_proc``), aggregates
  stats, then :func:`train` evolves the population.

Threading & processes
---------------------
- The DeSmuME emulator is created and used **inside each process** that runs it.
- The GTK main loop **must** run in a single process. This module designates one
  display-enabled process per batch as the *host* that creates the window and
  drives GTK via `GLib.timeout_add`.
- Overlays are computed by a **single background thread** (daemon) within each
  display-enabled process to keep the emulation loop responsive.

Shared-memory lifetime
----------------------
- Creation: Call :func:`safe_shared_memory` to create (or replace) a named
  shared-memory segment.
- Ownership: Workers **open** their frame buffer (``emu_frame_{id}``) by name
  and keep a persistent ``SharedMemory`` handle as long as they render frames.
- Teardown: After processes finish, the parent should **close and unlink**
  per-process frame segments to avoid resource-tracker warnings.

Examples
--------
Run 10 generations with a population of 32 where only one sample is displayed
each batch and overlays with IDs 0, 3, and 4 are enabled:

    >>> if __name__ == "__main__":
    ...     train(
    ...         num_iters=10,
    ...         pop_size=32,
    ...         show_samples=[False],   # broadcast later per batch
    ...         overlay_ids=[0, 3, 4],
    ...     )

Notes
-----
- This module expects the `mariokart_ds.nds` ROM to be available in the working
  directory and a valid savestate at index 3.
- ``on_draw_memoryview`` expects an emulator-provided RGBX memory buffer
  (4 bytes/pixel), and returns a premultiplied ARGB32 array suitable for Cairo.
- ``MODEL_KEY_MAP`` defines a simple thresholded policy:
  values >= 0.5 are pressed, and the accelerator button is always pressed when
  any action is taken.

"""

# Builtin dependencies
from __future__ import annotations
import random, math, os, sys, copy, time
from multiprocessing.managers import DictProxy
from multiprocessing import Process, Manager, Lock
from multiprocessing.shared_memory import SharedMemory
from queue import Queue
from threading import Thread
from typing import TypedDict, TypeAlias, cast
import argparse
from pathlib import Path

# External dependencies
from desmume.emulator import DeSmuME, SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_PIXEL_SIZE
from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer
from desmume.controls import Keys, keymask
from torch._prims_common import DeviceLikeType
import numpy as np
import gi

from src.main import SAVE_STATE_ID

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

# Local dependencies
from src.core.memory import *
from src.core.memory import read_clock
from src.core.model import Genome, EvolvedNet, load_genome, save_genome
from src.utils.vector import get_mps_device
from src.visualization.window import SharedEmulatorWindow, on_draw_memoryview
from src.visualization.overlay import AVAILABLE_OVERLAYS
from src.core.metric import DistanceMetric, FitnessScorer, Metric, collect_all, default_fitness_scorer

class EmulatorProcessConfig(TypedDict):
    id: int
    sample: Genome
    host: bool
    show: bool

class EmulatorBatchConfig(TypedDict):
    size: int
    display_shm_names: list[str]
    device: DeviceLikeType | None
    overlay_ids: list[int]
    metric_factories: list[Callable[[], Metric]]

class CheckpointRecord:
    id: int
    times: list[float]
    dists: list[float]

MODEL_KEY_MAP = {
    3: Keys.KEY_UP,
    2: Keys.KEY_DOWN,
    1: Keys.KEY_LEFT,
    0: Keys.KEY_RIGHT,
    4: Keys.KEY_B,
    5: Keys.KEY_A,
    6: Keys.KEY_X,
    7: Keys.KEY_Y,
    8: Keys.KEY_L,
    9: Keys.KEY_R,
    10: Keys.KEY_START,
    11: Keys.KEY_LEFT,
    12: Keys.KEY_RIGHT,
    13: Keys.KEY_UP,
    14: Keys.KEY_DOWN,
}

CHECKPOINT_DIR_PATH = "./model_checkpoints"

NUM_FRAMES_WITH_NOISE = 6500 # This is usually dependent on when (during the race) the savestate loads.

shm_names = []
best_genome: Genome | None = None

def safe_shared_memory(name: str, size: int):
    """Create or replace a named POSIX shared-memory segment.

    This helper guarantees that a shared-memory block with the given `name`
    exists with the requested `size`. If a stale block exists (e.g., from an
    earlier crashed run), it is closed and unlinked before creating a fresh one.

    Args:
      name: Symbolic name of the shared memory region
        (e.g., ``"emu_frame_0"``).
      size: Size in **bytes** to allocate for the region.

    Returns:
      multiprocessing.shared_memory.SharedMemory: An opened handle to the new
      shared-memory block. The caller owns the handle and is responsible for
      closing it (and unlinking at teardown time).

    Raises:
      ValueError: If `size <= 0`.
      OSError: If the OS cannot allocate or map the segment.

    Side Effects:
      - May unlink an existing segment of the same name.
      - Creates a new segment in the system shared-memory namespace.

    """
    from multiprocessing import shared_memory

    if size <= 0:
        raise ValueError("safe_shared_memory: size must be > 0")

    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        old = shared_memory.SharedMemory(name=name)
        old.close()
        old.unlink()
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    return shm


def initialize_emulator() -> DeSmuME:
    """Initialize and prime a DeSmuME emulator instance.

    Loads the MKDS ROM, restores a savestate (slot 3), mutes audio, and cycles
    once to ensure memory is initialized. Then spins until the emulator reports
    it is running.

    Returns:
      DeSmuME: A ready-to-use emulator instance positioned at the savestate.

    Notes:
      - This function blocks until ``emu.is_running()`` returns True.
      - The ROM path ``"mariokart_ds.nds"`` and savestate index are hard-coded.

    """
    emu = DeSmuME()
    emu.open("private/mariokart_ds.nds")
    emu.savestate.load(SAVE_STATE_ID)
    emu.volume_set(0)
    emu.cycle()
    os.system('clear')

    while not emu.is_running():
        print("Waiting for emulator...")

    return emu


def initialize_window(emu, config: EmulatorProcessConfig, batch_config: EmulatorBatchConfig) -> SharedEmulatorWindow | None:
    """Create and initialize the tiled GTK window for live visualization.

    Computes a near-square grid (``n_rows`` × ``n_cols``) based on the number
    of display-enabled processes, instantiates a renderer bound to `emu`, and
    returns a :class:`SharedEmulatorWindow` configured to read from the provided
    shared-memory frame names.

    Args:
      emu: Active :class:`DeSmuME` instance (used to build the renderer).
      display_count: Number of display-enabled workers to tile.
      shm_names: List of shared-memory segment names (``"emu_frame_{id}"``).

    Returns:
      SharedEmulatorWindow: GTK window object ready to be shown.

    Side Effects:
      - Initializes a GTK/Cairo renderer via :class:`AbstractRenderer`.

    """
    if not config['host']:
       return None

    shm_names = batch_config['display_shm_names']
    display_count = len(shm_names)
    width = 1000
    height = math.floor(width * (SCREEN_HEIGHT / SCREEN_WIDTH))
    n_cols = math.ceil(math.sqrt(display_count))
    n_rows = math.ceil(display_count / n_cols)
    renderer = AbstractRenderer.impl(emu)
    renderer.init()
    window = SharedEmulatorWindow(
        width=width,
        height=height,
        n_cols=n_cols,
        n_rows=n_rows,
        renderer=renderer,
        shm_names=shm_names,
    )
    return window


def initialize_overlays(
    config: EmulatorProcessConfig, batch_config: EmulatorBatchConfig
) -> Queue | None:
    """Start a background overlay thread and return its work queue.

    Given a list of overlay IDs, looks them up in :data:`AVAILABLE_OVERLAYS`,
    starts a single daemon thread that consumes :class:`DeSmuME` instances from
    a queue and applies the overlays. The queue is returned to the caller to
    submit per-frame overlay requests.

    Args:
      overlay_ids: List of overlay identifiers to enable (indexes into
        :data:`AVAILABLE_OVERLAYS`).
      device: Torch device on which overlay computations (if any) should run.

    Returns:
      Queue | None: If `overlay_ids` is non-empty, a ``Queue`` into which the
      caller should ``put(emu)`` once per frame, and ``put(None)`` on shutdown.
      Returns ``None`` when `overlay_ids` is empty.

    Notes:
      - The overlay worker catches exceptions per overlay and propagates a
        summarized error message on failure via :func:`safe_thread`.
      - Overlays are executed off the emulation thread to avoid jitter.

    """
    overlay_ids = batch_config['overlay_ids']
    if not config['show'] or not len(overlay_ids) > 0:
        return None

    device = batch_config['device']
    overlays = []
    for id in overlay_ids:
        overlays.append(AVAILABLE_OVERLAYS[id])

    emu_queue = Queue()

    def worker():
        nonlocal overlays, emu_queue, id
        assert emu_queue is not None
        while True:
            emu_instance = emu_queue.get()
            if emu_instance is None:
                break
            for overlay in overlays:
                safe_overlay = safe_thread(overlay, proc_id=id)
                safe_overlay(emu_instance, device=device)

            emu_queue.task_done()

    thread = Thread(target=worker, daemon=True)
    thread.start()

    return emu_queue


def handle_controls(emu: DeSmuME, logits: torch.Tensor, max_frame_with_noise: int = 0):
    """Apply model outputs to emulator controls with a simple threshold policy.

    All values ``>= 0.5`` are considered pressed for the corresponding
    ``MODEL_KEY_MAP`` entry. Additionally, when any action is pressed, the
    accelerator (mapped to ``MODEL_KEY_MAP[5]``) is also pressed to keep the
    kart moving.

    Args:
      emu: Active emulator instance whose keypad state will be updated.
      logits: 1D tensor of action activations aligned with ``MODEL_KEY_MAP``.

    Side Effects:
      - Calls ``emu.input.keypad_update(0)`` and
        ``emu.input.keypad_add_key(...)`` multiple times.

    """
    logits_list = logits.tolist()
    emu.input.keypad_update(0)
    for i, v in enumerate(logits_list):
        if read_clock(emu) < max_frame_with_noise:
            v += random.random()
        if v < 0.5:
            continue

        emu.input.keypad_add_key(keymask(MODEL_KEY_MAP[i]))

    emu.input.keypad_add_key(keymask(MODEL_KEY_MAP[5]))





def initialize_model(emu: DeSmuME, config: EmulatorProcessConfig, batch_config: EmulatorBatchConfig):
    device = batch_config['device']
    sample = config['sample']
    model = EvolvedNet(sample, device=device)
    forward = get_forward_func(emu, model, device=device)
    return forward

def _run_process(
    training_stats: dict[int, dict[str, float]],
    training_stats_lock, 
    config: EmulatorProcessConfig,
    batch_config: EmulatorBatchConfig
):
    assert config['show'] if config['host'] else True, "Host processes must have display enabled"
    

    # Initialize emulator
    emu = initialize_emulator()

    # Initialize model
    forward = initialize_model(emu, config, batch_config)
    metrics: list[Metric] = [b() for b in batch_config['metric_factories']]
    for m in metrics: m.reset()
    device = batch_config['device']

    # Initialize display shared memory buffer as numpy array
    frame = None
    if config["show"]:
        id = config['id']
        shm_frame = SharedMemory(name=f"emu_frame_{id}")
        frame = np.ndarray(
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4),
            dtype=np.uint8,
            buffer=shm_frame.buf,
        )

    # Initialize window
    window = initialize_window(emu, config, batch_config)

    # Set overlay overlay thread
    emu_queue = initialize_overlays(config, batch_config)

    def tick():
        nonlocal stats, emu_queue, frame, emu, window
        emu.cycle()

        if frame is not None:
            # Copy display data to shared memory buffer
            buf = emu.display_buffer_as_rgbx()[: SCREEN_PIXEL_SIZE * 4]
            new_frame = on_draw_memoryview(buf, SCREEN_WIDTH, SCREEN_HEIGHT, 1.0, 1.0)
            np.copyto(frame, new_frame)


        # Inference / Game Update
        logits = forward()
        for metric in metrics:
            metric.update(emu, device=device)
            
        if logits is None:
            id = config['id']
            send_window_end_signal(id)
            return False


        handle_controls(emu, logits, NUM_FRAMES_WITH_NOISE)
        if emu_queue is not None:
            # Queue Overlay Request
            emu_queue.put(emu)

        return True

    while not config['host']:
        val = tick()
        if val == False:
            break


    if window is not None:
        # Will incrementally check if the population has died
        def check_end():
            """Quit GTK when all visible frames are cleared to zeros."""
            for name in batch_config['display_shm_names']:
                shm = SharedMemory(name=name)
                arr = np.ndarray(
                    (SCREEN_WIDTH, SCREEN_HEIGHT, 4), dtype=np.uint8, buffer=shm.buf
                )

                if arr.sum() != 0:
                    return True

            Gtk.main_quit()
            return False

        GLib.timeout_add(200, check_end)  # non-blocking
        GLib.timeout_add(1, tick)  # non-blocking
        window.show_all()
        window.set_keep_above(True)   # Keeps window above others temporarily
        window.present_with_time(Gtk.get_current_event_time())  # More reliable focus timing
        Gtk.main()  # blocking

    # Safe thread shutdown for overlay
    if emu_queue is not None:
        emu_queue.put(None)


    # Log results
    with training_stats_lock:
        id = config['id']
        stats = collect_all(metrics)
        training_stats[id] = stats


def safe_thread(func, proc_id, thread_id=0):
    """Wrap a function for background execution with nicer error reporting.

    The returned wrapper calls `func(*args, **kwargs)` and converts any
    exception into a concise message identifying the logical process and thread
    of origin.

    Args:
      func: Callable to wrap.
      proc_id: Integer process identifier for error messages.
      thread_id: Integer thread identifier for error messages.

    Returns:
      Callable: A new callable with identical signature that raises a concise
      :class:`Exception` on failure.

    """
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            raise Exception(f"Error on thread {thread_id} of process {proc_id}")

    return wrapper


def send_window_end_signal(id):
    """Zero a per-process frame buffer to signal the host window to exit.

    Args:
      id: Process index whose frame buffer should be cleared.

    Side Effects:
      - Writes zeros into the shared frame ``emu_frame_{id}``, which is used by
        the host window's polling logic to detect end-of-batch.

    """
    shm = SharedMemory(name=f"emu_frame_{id}")
    arr = np.ndarray((SCREEN_HEIGHT, SCREEN_WIDTH, 4), dtype=np.uint8, buffer=shm.buf)
    arr[:] = 0.0


def get_forward_func(emu: DeSmuME, model: EvolvedNet, device: DeviceLikeType | None = None):
    """Build a closure that performs one model step.

    The returned callable reads emulator memory for sensor inputs, constructs the
    model input vector, computes the control logits. When a terminal condition is reached (e.g.
    clock > 10000), it returns a NoneType value instead of logits.

    Args:
      emu: Active emulator instance to read game state from.
      model: Evolved network to evaluate (expects 6 inputs → action logits).
      device: Torch device on which tensors are constructed and the model runs.

    Returns:
      Callable[[], torch.Tensor | None]:
        A no-argument function that returns either a 1D tensor of action logits
        or NoneType signaling the end of this individual's run.

    Sensor model:
      - Distances: forward/left/right obstacle distances are read and mapped
        through ``tanh(1 - d / s1)`` with ``s1 = 60.0`` to compress range.
      - Angles: direction to the next checkpoint as (cos θ, sin θ, -sin θ).

    Notes:
      - Checkpoint bookkeeping appends tuples of ``(delta_time, distance_at_split)``.
      - This function reads directly from emulator memory via utility helpers.

    """
    

    def forward() -> torch.Tensor | None:
        clock = read_clock(emu)
        if clock > 20000:
            return None

        s1 = 1000.0

        # Sensor inputs (obstacle distances)
        forward_d = read_forward_distance_obstacle(emu, device=device, interval=(-0.1, 0.1))
        left_d = read_left_distance_obstacle(emu, device=device, interval=(-0.5, 0.5))
        right_d = read_right_distance_obstacle(emu, device=device, interval=(-0.5, 0.5))
        inputs_dist1 = torch.tensor([forward_d, left_d, right_d], device=device)
        inputs_dist1 = inputs_dist1 < 200
        
        # Angular relationship to next checkpoint
        angle = read_direction_to_checkpoint(emu, device=device)
        forward_a = torch.cos(angle)
        left_a = torch.sin(angle)
        right_a = -torch.sin(angle)
        inputs_dist2 = torch.tensor([forward_a, left_a, right_a], device=device)

        # Model inference
        inputs = torch.cat([inputs_dist1, inputs_dist2])
        
        logits = model(inputs_dist1)

        return logits

    return forward


def run_training_batch(
    batch_population: list[Genome],
    show_samples: list[bool],
    training_stats: DictProxy[int, dict[str, float]],
    training_stats_lock,
    batch_config: EmulatorBatchConfig
):
    """Evaluate a batch of genomes concurrently, optionally with live display.

    One process in the batch is promoted to the *display host* (first True in
    `show_samples`) and creates a tiled GTK window that reads from the
    per-process shared-memory frames listed in `shm_names`. Additional True
    entries run as display workers; False entries run headless.

    Args:
      batch_pop: Slice of the population to evaluate in this batch.
      show_samples: Per-individual flags controlling display mode; exactly one
        True is chosen as the host (the first True), additional Trues are
        workers; all False means fully headless batch.
      overlay_ids: Enabled overlay identifiers.
      lock: IPC lock for synchronized writes to `pop_stats`.
      pop_stats: Manager dict where each process writes a stats dict under its
        local batch index.

    Side Effects:
      - Creates per-process shared-memory frame segments for display-enabled
        individuals.
      - Spawns processes with appropriate targets and joins them.

    """
    batch_size = batch_config['size']

    processes = []

    host_proc_found = False
    for id, sample, show in zip(range(batch_size), batch_population, show_samples):
        config: EmulatorProcessConfig = {
            "id": id,
            "sample": sample,
            "host": False,
            "show": show
        }

        if not host_proc_found:
            host_proc_found = True
            config["host"] = True

        if show:
            shm_frame = SharedMemory(name=f"emu_frame_{id}")
            frame = np.ndarray(
                shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4),
                dtype=np.uint8,
                buffer=shm_frame.buf,
            )
            frame[:] = 1.0

        process = Process(
            target=_run_process,
            args=(training_stats, training_stats_lock, config, batch_config),
            daemon=True,
        )
        processes.append(process)

    # Start processes
    for p in processes:
        p.start()

    # Join processes
    for p in processes:
        p.join()


def run_training_session(
    pop: list[Genome],
    num_proc: int | None = None,
    show_samples: list[bool] = [True],
    overlay_ids: list[int] = [],
    metric_factories: list[Callable[[], Metric]] = [],
    device: DeviceLikeType | None = None
) -> dict[int, dict[str, float]]:
    """Evaluate the full population in parallel batches and collect statistics.

    The population is partitioned into batches of size ``min(num_proc, remaining)``.
    Each batch is launched via :func:`run_training_batch`, returning when all
    processes in the batch complete and their stats have been merged.

    Args:
      pop: Full population of genomes to evaluate.
      num_proc: Maximum number of concurrent processes. If ``None``, uses
        ``os.cpu_count() - 1``.
      show_samples: List of booleans determining which individuals in each batch
        should display; a one-element list (e.g., ``[False]``) is **broadcast**
        to the batch size on each iteration.
      overlay_ids: Overlay identifiers to enable in display-enabled processes.

    Returns:
      dict[int, dict[int, list[tuple[float, float]]]]: Mapping of *global*
      population index to that individual's checkpoint stats dict.

    Notes:
      - This function uses a :class:`multiprocessing.Manager` ``dict`` so that
        per-process stats can be retrieved without explicit pipes or queues.
      - Shared-memory frame buffers are currently **not** unlinked here; consider
        cleaning them in a higher-level teardown if needed.

    """
    global shm_names

    # If no number of processes is specified, then we'll use the max minus one
    if num_proc is None:
        num_proc = os.cpu_count()
        assert num_proc is not None
        num_proc -= 1

    if len(show_samples) == 1:
        show_samples *= num_proc

    pop_stats: dict[int, dict[str, float]] = {}
    pop_size = len(pop)
    count = 0

    shm_names = []
    for i in range(num_proc):
        if not show_samples[i]: continue
        name = f"emu_frame_{i}"
        size = SCREEN_HEIGHT * SCREEN_WIDTH * 4
        shm_frame = safe_shared_memory(name=name, size=size)
        frame = np.ndarray(
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 4),
            dtype=np.uint8,
            buffer=shm_frame.buf,
        )
        shm_names.append(name)

    while count < pop_size:
        batch_size = min(pop_size - count, num_proc)

        batch_config: EmulatorBatchConfig = {
            "overlay_ids": overlay_ids,
            "display_shm_names": shm_names,
            "size": batch_size,
            "metric_factories": metric_factories,
            "device": device
        }

        show_samples = show_samples[:batch_size]

        with Manager() as manager:
            # Create shared list for stats (locking)
            shared_pop_stats: DictProxy[int, dict[str, float]] = manager.dict()
            lock = Lock()

            run_training_batch(
                pop[count : count + batch_size],
                show_samples=show_samples,
                training_stats=shared_pop_stats,
                training_stats_lock=lock,
                batch_config=batch_config
            )

            for k, s in shared_pop_stats.items():
                pop_stats[count + k] = s

        count += batch_size

    # TODO: Cleanup all shared memory buffers here
    return pop_stats


def fitness(pop_stats: dict[int, dict[str, float]], scorer: FitnessScorer) -> list[float]:
    """Compute scalar fitness from per-checkpoint stats.

    The current fitness heuristic sums the recorded distances across all
    checkpoint splits for each individual.

    Args:
      pop_stats: Mapping of population index to that individual's stats dict
        (``checkpoint_id -> [(delta_time, distance_at_split), ...]``).

    Returns:
      list[float]: Fitness values *ordered by population index*.

    """
    scores = []
    for k in sorted(pop_stats.keys()):
        metrics = pop_stats[k]
        score = scorer(metrics)
        scores.append(score)
        
    return scores



def train(
    num_iters: int,
    pop_size: int,
    log_interval: int = 1,
    top_k: int | float = 0.1,
    device: DeviceLikeType | None = None,
    scorer: FitnessScorer = default_fitness_scorer,
    load_checkpoint_path: str | Path | None = None,
    
    **simulation_kwargs,
):
    """Main evolutionary training loop (selection + mutation).

    Repeats:
      1) Evaluate the current population via :func:`run_training_session`.
      2) Rank by fitness.
      3) Keep the best, and refill the population by mutating uniformly sampled
         parents from the top-k set.

    Args:
      num_iters: Number of generations to run.
      pop_size: Number of individuals per generation.
      log_interval: Print progress every N generations.
      top_k: Either the number of top individuals to sample parents from, or a
        fraction in ``(0, 1]`` interpreted as a proportion of the population.
      **simulation_kwargs: Passed through to :func:`run_training_session`.

    Side Effects:
      - Prints best fitness per `log_interval`.
      - Mutates and replaces the population in place each generation.

    """
    global best_genome
    if load_checkpoint_path is None:
        pop: list[Genome] = [Genome(3, 2, device=device) for _ in range(pop_size)]
    else:
        pop: list[Genome] = [load_genome(load_checkpoint_path) for _ in range(pop_size)]
    
    for g in pop: g.mutate_add_conn()  # start with random links

    if top_k <= 1:
        top_k = int(round(len(pop) * top_k))
    assert isinstance(top_k, int), "top_k must be an integer or a float less than 1"

    for n in range(num_iters):
        stats = run_training_session(pop, device=device, **simulation_kwargs)
        
        scores = fitness(stats, scorer)
        scores = sorted(enumerate(scores), reverse=True, key=lambda x: x[1])
        
        print(best_genome.conns if best_genome is not None else "")

        if n % log_interval == 0:
            os.system("clear")
            print(f"Best Fitness: {scores[0][1]}")

        first_i = scores[0][0]
        best_genome = copy.deepcopy(pop[first_i])
        newpop = [copy.deepcopy(pop[first_i])]
        for _ in range(len(pop) - 1):
            rand_i = random.choice(scores[:5])[0]
            g = copy.deepcopy(pop[rand_i])
            random.choice([g.mutate_weight, g.mutate_add_conn, g.mutate_add_node])()
            newpop.append(g)
        pop = newpop

def make_distance_metric():
    return DistanceMetric()

def main():
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--pop_size", type=int, default=-1)
    parser.add_argument("--num_processes", type=int)
    parser.add_argument("--num_displays", type=int, default=0)
    parser.add_argument("--load_checkpoint_path", type=str)
    parser.add_argument("--load_checkpoint_recent", type=bool, default=False)
    parser.add_argument("--save_checkpoint", type=bool, default=False)
    args = parser.parse_args()
    
    num_proc = args.num_processes
    if num_proc is None:
        num_proc = os.cpu_count()
        if num_proc is not None:
            num_proc -= 1
        else:
            num_proc = 1
    
    num_displays = args.num_displays
    if num_displays < 0:
        num_displays = num_proc
    
    num_headless = num_proc - num_displays
    
    device = get_mps_device()
    
    pop_size = args.pop_size
    if pop_size < 0:
        pop_size = num_proc
        
    load_checkpoint_path: str | Path | None = None
    if args.load_checkpoint_recent == True:
        path = Path(CHECKPOINT_DIR_PATH)
        
        files = sorted(
            path.iterdir(),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        load_checkpoint_path = files[0]
    elif args.load_checkpoint_path is not None:
        load_checkpoint_path = args.load_checkpoint_path
        
    
    show_displays = [True] * num_displays + [False] * num_headless
    
    try:
        
        train(
            num_iters=args.num_iters,
            pop_size=pop_size,
            device=device,
            top_k=args.top_k,
            show_samples=show_displays,
            overlay_ids=[0, 3, 4],
            num_proc=num_proc,
            metric_factories=[make_distance_metric],
            load_checkpoint_path=load_checkpoint_path
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        global shm_names, best_genome
        for name in shm_names:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        
        if args.save_checkpoint == True and best_genome is not None:
            dest_file_path = Path(CHECKPOINT_DIR_PATH).joinpath(f"chk_{str(hash(best_genome))[:8]}.json")
            save_genome(best_genome, dest_file_path)
            print(f"Model Weights Saved at {dest_file_path}")

if __name__ == "__main__":
    main()
    