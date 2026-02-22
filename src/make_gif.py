import sys
import glob
import os
from PIL import Image


def make_gif(pattern, output, fps):
    files = sorted(
        glob.glob(pattern), key=lambda x: int("".join(filter(str.isdigit, x)))
    )
    if not files:
        print(f"No files found for pattern {pattern}")
        return

    images = [Image.open(f) for f in files]

    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"GIF saved as {output}, {len(images)} frames at {fps} fps")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make a GIF from PNGs")
    parser.add_argument(
        "--pattern",
        default="../RESULTS/pp_step_*.png",
        help="Glob pattern to search for PNGs",
    )
    parser.add_argument("--output", default="vid_of_pp.gif", help="Output GIF filename")
    parser.add_argument("--fps", type=float, default=5, help="Frames per second")
    args = parser.parse_args()

    make_gif(args.pattern, args.output, args.fps)
