import argparse
import os
import numpy as np

from tqdm import trange

from models import Lorenz

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="directory to store the output dataset")
    parser.add_argument("--sigma", type=float, default=10,
                        help="sigma parameter for the lorenz system")
    parser.add_argument("--rho", type=float, default=28,
                        help="rho parameter for the lorenz system")
    parser.add_argument("--beta", type=float, default=8/3,
                        help="beta parameter for the lorenz system")
    parser.add_argument("--train-noise", type=float, default=8e2,
                        help="noise sigma to inject as train data input")
    parser.add_argument("--train-sim-time", type=float, default=100,
                        help="how long to simulate for training set in [s]")
    parser.add_argument("--val-sim-time", type=float, default=30,
                        help="how long to simulate for validation set in [s]")
    parser.add_argument("--val-sin-mag", type=float, default=5e1,
                        help="sinusoidal input magnitude for validation data")
    parser.add_argument("--val-sin-period", type=float, default=.3,
                        help="sinusoidal input period for validation data in [s]")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="dt for integration in [s]")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for np.random")
    return parser.parse_args()

def generate_data(lorenz, args, validation=False):
    # random initial condition
    lorenz.set_states(x=1., y=1., z=20.)

    if validation:
        steps = int(args.val_sim_time / args.dt)
    else:
        steps = int(args.train_sim_time / args.dt)

    # simulate lorenz system
    states = lorenz.get_states_array()

    ret = np.zeros((), dtype=[
        ("state", "<f4", (steps, states.shape[0])),
        ("dstate", "<f4", (steps, states.shape[0])),
        ("control", "<f4", (steps, 1)), # u is the only one input
    ])
    for i in trange(steps):
        if validation:
            u = np.sin(i * args.dt * 2 * np.pi / args.val_sin_period) * args.val_sin_mag
        else:
            u = np.random.randn() * args.train_noise

        # save current state and input
        ret["state"][i] = states
        ret["control"][i] = [u]
        dstate_dict = lorenz.dynamics({"u": u})
        ret["dstate"][i] = [dstate_dict["x"], dstate_dict["y"], dstate_dict["z"]]

        # step through simulation
        lorenz.step({"u": u}, args.dt)
        states = lorenz.get_states_array()

    return ret

if __name__ == "__main__":
    args = parse_args()

    # set random seed
    np.random.seed(args.seed)

    # generate dataset
    lorenz = Lorenz(args.sigma, args.rho, args.beta)
    train_dataset = generate_data(lorenz, args, False)
    val_dataset = generate_data(lorenz, args, True)

    # save dataset to disk
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "train"), train_dataset)
    np.save(os.path.join(args.output_dir, "val"), val_dataset)
