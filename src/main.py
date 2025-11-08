from __future__ import annotations
import cairo
import threading
import queue
import torch
import os
import gi
from pynput import keyboard
from desmume.emulator import DeSmuME
from desmume.controls import Keys, keymask
#from private.mkds_python_bindings.testing.driver import driver_t
#from private.mkds_python_bindings.testing.nnsfnd import NNSFndList
#from private.mkds_python_bindings.testing.sfx import sfx_emitter_t
#from private.mkds_python_bindings.testing.list import list_link_t
from src.visualization.draw import consume_draw_stack, draw_text, draw_paragraph
from src.core.memory import *
from src.utils.vector import get_mps_device
from src.visualization.overlay import (
    distance_overlay,
    player_overlay,
    raycasting_overlay,
    collision_overlay,
    checkpoint_overlay_1,
    checkpoint_overlay_2,
)
import ctypes

os.environ["PKG_CONFIG_PATH"] = "/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
    "/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"
)

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

# ----------------------------
# CONSTANTS
# ----------------------------
#device = get_mps_device()
device = None
SCALE = 3
SCREEN_WIDTH, SCREEN_HEIGHT = 256, 192
SAVE_STATE_ID = 0


# ----------------------------
# GLOBAL VARIABLES
# ----------------------------
input_state = set()
is_running = False
scene_next = None
renderer = None
emu_global: DeSmuME | None = None
callback = None
worker_queue: queue.Queue[DeSmuME | None] = queue.Queue()

# ----------------------------
# KEY MAPPING
# ----------------------------
KEY_MAP = {
    "w": Keys.KEY_UP,
    "s": Keys.KEY_DOWN,
    "a": Keys.KEY_LEFT,
    "d": Keys.KEY_RIGHT,
    "z": Keys.KEY_B,
    "x": Keys.KEY_A,
    "u": Keys.KEY_X,
    "i": Keys.KEY_Y,
    "q": Keys.KEY_L,
    "e": Keys.KEY_R,
    " ": Keys.KEY_START,
    "left": Keys.KEY_LEFT,
    "right": Keys.KEY_RIGHT,
    "up": Keys.KEY_UP,
    "down": Keys.KEY_DOWN,
}


# ----------------------------
# ASYNC KEYBOARD HANDLER
# ----------------------------
def start_keyboard_listener():
    """Starts a non-blocking keyboard listener in a separate thread."""

    def on_press(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            if name in KEY_MAP:
                input_state.add(name)
        except Exception:
            pass

    def on_release(key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name
            if name in KEY_MAP:
                input_state.discard(name)
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


# ----------------------------
# DRAW CALLBACK
# ----------------------------
overlay_surface_cache: cairo.Surface | None = None


def on_draw_main(widget: Gtk.DrawingArea, ctx: cairo.Context):
    global renderer, scene_current, emu_global, overlay_surface_cache
    if renderer is None:
        return False

    ctx.scale(SCALE, SCALE)
    renderer.screen(SCREEN_WIDTH, SCREEN_HEIGHT, ctx, 0)

    src_surface: cairo.ImageSurface = ctx.get_target()

    overlay_surface = src_surface.create_similar(
        cairo.CONTENT_COLOR_ALPHA, SCREEN_WIDTH, SCREEN_HEIGHT
    )

    overlay_surface.set_device_scale(*src_surface.get_device_scale())

    overlay_ctx = cairo.Context(overlay_surface)
    overlay_ctx.set_operator(cairo.OPERATOR_CLEAR)
    overlay_ctx.paint()
    overlay_ctx.set_operator(cairo.OPERATOR_OVER)
    overlay_ctx.set_antialias(cairo.ANTIALIAS_NONE)

    num_calls = consume_draw_stack(overlay_ctx)

    if num_calls == 0 and overlay_surface_cache is not None:

        to_paint = overlay_surface_cache
    else:

        overlay_surface_cache = overlay_surface
        to_paint = overlay_surface

    pattern = cairo.SurfacePattern(to_paint)
    pattern.set_filter(cairo.FILTER_NEAREST)
    ctx.set_source(pattern)
    ctx.set_operator(cairo.OPERATOR_OVER)

    
    ctx.paint()
    return False


def on_configure_main(widget: Gtk.DrawingArea, *args):
    global renderer
    if renderer:
        renderer.reshape(widget, 0)
    return True


# ----------------------------
# GTK WINDOW
# ----------------------------
class EmulatorWindow(Gtk.Window):
    def __init__(self):
        super().__init__(title="MarI/O Kart")
        self.set_default_size(SCREEN_WIDTH, SCREEN_HEIGHT)
        drawing_area = Gtk.DrawingArea()
        drawing_area.set_size_request(SCREEN_WIDTH * SCALE, SCREEN_HEIGHT * SCALE)
        drawing_area.connect("draw", on_draw_main)
        drawing_area.connect("configure-event", on_configure_main)
        self.add(drawing_area)
        self.drawing_area = drawing_area
        self.connect("destroy", Gtk.main_quit)
        self.set_events(Gdk.EventMask.KEY_PRESS_MASK | Gdk.EventMask.KEY_RELEASE_MASK)


# ----------------------------
# EMULATION WORKER
# ----------------------------
def worker():
    global scene_next, callback, is_running
    assert callback is not None
    while True:
        emu_instance = worker_queue.get()
        if emu_instance is None:
            break
        try:
            callback(emu_instance)
        except Exception as e:
            raise RuntimeError(
                f"error on thread {threading.current_thread().name}: {e}"
            )

        worker_queue.task_done()
    pass


# ----------------------------
# MAIN LOOP
# ----------------------------
def run_emulator(overlays):
    global renderer, callback, emu_global, is_running
    emu = DeSmuME()
    emu.open("private/mariokart_ds.nds")
    emu.savestate.load(1)

    emu_global = emu
    emu.volume_set(0)

    n = load_current_nkm(emu, device=device)
    k = load_current_kcl(emu, device=device)

    from desmume.frontend.gtk_drawing_area_desmume import AbstractRenderer

    renderer = AbstractRenderer.impl(emu)
    renderer.init()

    window = EmulatorWindow()
    window.show_all()

    def main_callback(emu: DeSmuME):
        global device
        for overlay in overlays:
            overlay(emu, device)

    callback = main_callback
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    is_running = True

    def tick():
        global scene_current, scene_next
        emu.cycle()

        emu.input.keypad_update(0)
        for key in input_state:
            emu.input.keypad_add_key(keymask(KEY_MAP[key]))
        
        if not is_running:
            Gtk.main_quit()

        window.drawing_area.queue_draw()
        worker_queue.put(emu)
        return True

    # run at 60fps, non-blocking
    GLib.timeout_add(16, tick)
    Gtk.main()
    window.connect("destroy", lambda w: worker_queue.put(None))


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    device = get_mps_device()
    start_keyboard_listener()
    run_emulator(
        [
            collision_overlay,
            checkpoint_overlay_1,
            checkpoint_overlay_2,
            distance_overlay
        ],
    )
