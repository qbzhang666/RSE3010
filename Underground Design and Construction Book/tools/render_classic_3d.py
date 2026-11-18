"""Three-dimensional figures of the book, built procedurally in Blender.

Every scene is a module tools/render3d/scene_<name>.py with
    NAME, CHAPTER, DATA_KEY (key in figures/render/scenes.json, or None),
    DEFAULTS (parameters from the book's worked examples), TARGET (look-at point),
    build(d) that makes the geometry, materials, camera and lights.
No model files are needed and no text is baked into a render; the caption carries it.

Run headless (locally for a check, on the M3 GPUs for the final stills):

    blender -b --factory-startup -P tools/render_classic_3d.py -- --list
    blender -b --factory-startup -P tools/render_classic_3d.py -- \
        --scene plastic_zone --out figures/render --samples 32 --res 1200x800 --device CPU
    blender -b --factory-startup -P tools/render_classic_3d.py -- \
        --scene all --out figures/render --samples 1024 --res 3600x2400 --device CUDA
    # 360 degree turntable for lectures: frames out/<name>_turn/f_0001.png ...
    blender -b --factory-startup -P tools/render_classic_3d.py -- \
        --scene cavern_sequence --turntable 240 --res 1920x1080 --samples 256 --device CUDA
"""

import argparse
import importlib
import json
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_DIR = os.path.join(HERE, "render3d")
sys.path.insert(0, SCENE_DIR)


def discover():
    mods = {}
    for f in sorted(os.listdir(SCENE_DIR)):
        if f.startswith("scene_") and f.endswith(".py"):
            m = importlib.import_module(f[:-3])
            mods[m.NAME] = m
    return mods


def setup_render(res, samples, device):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    w, hgt = (int(v) for v in res.lower().split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, hgt
    sc.render.resolution_percentage = 100
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.film_transparent = True               # composited onto white by flatten_white()
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"
    if device != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = device
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use = dev.type == device
        sc.cycles.device = "GPU"


def flatten_white(path):
    """Composite the transparent render onto white, so the page shows no grey frame."""
    import numpy as np
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    al = px[:, 3:4]
    px[:, :3] = px[:, :3] * al + (1.0 - al)
    px[:, 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def turntable(target, frames, outdir):
    """Orbit the scene camera once about a vertical axis through the target."""
    sc = bpy.context.scene
    cam = sc.camera
    pivot = bpy.data.objects.new("pivot", None)
    bpy.context.collection.objects.link(pivot)
    pivot.location = target
    mw = cam.matrix_world.copy()
    cam.parent = pivot
    cam.matrix_parent_inverse = pivot.matrix_world.inverted()
    cam.matrix_world = mw
    os.makedirs(outdir, exist_ok=True)
    for k in range(frames):
        pivot.rotation_euler = (0.0, 0.0, 2 * math.pi * k / frames)
        bpy.context.view_layer.update()
        sc.render.filepath = os.path.join(outdir, f"f_{k + 1:04d}.png")
        if os.path.exists(sc.render.filepath):
            continue                                # resumable after a time-out
        bpy.ops.render.render(write_still=True)
        flatten_white(sc.render.filepath)
        print(f"frame {k + 1}/{frames}", flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="all", help="name, comma list, or all")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "figures", "render", "scenes.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures", "render"))
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", default="1500x1000")
    ap.add_argument("--device", default="CPU", choices=["CPU", "CUDA", "OPTIX"])
    ap.add_argument("--turntable", type=int, default=0, help="number of frames for a 360 degree orbit")
    ap.add_argument("--save-blend", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    scenes = discover()
    if args.list:
        for n, m in scenes.items():
            print(f"{n:24s} {m.CHAPTER}")
        return
    data = json.load(open(args.data, encoding="utf-8")) if os.path.exists(args.data) else {}
    names = list(scenes) if args.scene == "all" else args.scene.split(",")
    os.makedirs(args.out, exist_ok=True)
    for name in names:
        m = scenes[name]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        import common
        common.reset_materials()
        d = dict(m.DEFAULTS)
        if m.DATA_KEY:
            d.update(data[m.DATA_KEY])
        m.build(d)
        setup_render(args.res, args.samples, args.device)
        if args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(args.out, name + ".blend")))
        if args.turntable:
            turntable(m.TARGET, args.turntable, os.path.abspath(os.path.join(args.out, name + "_turn")))
            print(f"turntable {name} done", flush=True)
            continue
        bpy.context.scene.render.filepath = os.path.abspath(os.path.join(args.out, name + ".png"))
        bpy.ops.render.render(write_still=True)
        flatten_white(bpy.context.scene.render.filepath)
        print(f"rendered {name} -> {bpy.context.scene.render.filepath}", flush=True)


if __name__ == "__main__":
    main()
