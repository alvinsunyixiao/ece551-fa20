import argparse
import os
import numpy as np

from tqdm import trange

from models import Pendulum

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="directory to store the output dataset")
    parser.add_argument("-l", "--length", type=float, default=0.4,
                        help="length of the pendulum in [m]")
    parser.add_argument("-m", "--mass", type=float, default=1.0,
                        help="mass of the pendulum in [kg]")
    parser.add_argument("-d", "--damping", type=float, default=1e-1,
                        help="damping of the pendulum in [N M / (rad / s)]")
    parser.add_argument("--train-noise", type=float, default=8e1,
                        help="noise sigma to inject as train data input")
    parser.add_argument("--train-sim-time", type=float, default=100,
                        help="how long to simulate for training set in [s]")
    parser.add_argument("--val-sim-time", type=float, default=30,
                        help="how long to simulate for validation set in [s]")
    parser.add_argument("--val-sin-mag", type=float, default=1e1,
                        help="sinusoidal input magnitude for validation data")
    parser.add_argument("--val-sin-period", type=float, default=.5,
                        help="sinusoidal input period for validation data in [s]")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="dt for integration in [s]")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for np.random")
    return parser.parse_args()

def generate_data(pendulum, args, validation=False):
    # random initial condition
    pendulum.set_states(theta=np.random.uniform(0, 2 * np.pi), theta_dot=0.)

    if validation:
        steps = int(args.val_sim_time / args.dt)
    else:
        steps = int(args.train_sim_time / args.dt)

    # simulate pendulum
    states = pendulum.get_states_array()

    ret = np.zeros((), dtype=[
        ("state", "<f4", (steps, states.shape[0])),
        ("dstate", "<f4", (steps, states.shape[0])),
        ("control", "<f4", (steps, 1)), # tau is the only one input
    ])
    for i in trange(steps):
        if validation:
            tau = np.sin(i * args.dt * 2 * np.pi / args.val_sin_period) * args.val_sin_mag
        else:
            tau = np.random.randn() * args.train_noise

        # save current state and input
        ret["state"][i] = states
        ret["control"][i] = [tau]
        dstate_dict = pendulum.dynamics({"tau": tau})
        ret["dstate"][i] = [dstate_dict["theta"], dstate_dict["theta_dot"]]

        # step through simulation
        pendulum.step({"tau": tau}, args.dt)
        states = pendulum.get_states_array()

    return ret

if __name__ == "__main__":
    args = parse_args()

    # set random seed
    np.random.seed(args.seed)

    # generate dataset
    pendulum = Pendulum(args.length, args.mass, args.damping)
    train_dataset = generate_data(pendulum, args, False)
    val_dataset = generate_data(pendulum, args, True)

    # save dataset to disk
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "train"), train_dataset)
    np.save(os.path.join(args.output_dir, "val"), val_dataset)
