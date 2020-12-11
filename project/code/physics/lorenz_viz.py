import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from physics.models import Lorenz

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="path to the input dataset")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="dt for integration in [s] (default to 1e-3)")
    parser.add_argument("--interval", type=int, default=20,
                        help="how many frames per animation (default to 20)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    lorenz = Lorenz(0, 0, 0)
    data = np.load(args.input)

    def animate_func(i):
        idx = i * args.interval;
        lorenz.set_states(x=data["state"][idx, 0],
                          y=data["state"][idx, 1],
                          z=data["state"][idx, 2])
        return lorenz.draw()

    lorenz.init_draw()
    anim = FuncAnimation(lorenz.fig, func=animate_func, interval=args.interval * args.dt * 1e3,
                         blit=True, frames=int(data["state"].shape[0] / args.interval))
    plt.show()
