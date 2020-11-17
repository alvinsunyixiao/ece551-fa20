import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from physics.models import Pendulum

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
    pend = Pendulum(0.1, 0.01, 1e-4)
    data = np.load(args.input)

    def animate_func(i):
        idx = i * args.interval;
        pend.set_states(theta=data["state"][idx, 0], theta_dot=data["state"][idx, 1])
        return pend.draw()

    pend.init_draw()
    anim = FuncAnimation(pend.fig, func=animate_func, interval=args.interval * args.dt * 1e3,
                         blit=True, frames=int(data["state"].shape[0] / args.interval))
    plt.show()
