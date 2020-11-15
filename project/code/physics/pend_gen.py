import argparse
import os
import numpy as np

from tqdm import trange

from models import Pendulum

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="directory to store the output dataset")
    parser.add_argument("-l", "--length", type=float, default=0.1,
                        help="length of the pendulum in [m] (default to 0.1)")
    parser.add_argument("-m", "--mass", type=float, default=0.01,
                        help="mass of the pendulum in [kg] (default to 0.01)")
    parser.add_argument("-d", "--damping", type=float, default=1e-4,
                        help="damping of the pendulum in [N M / (rad / s)] (default to 1e-4)")
    parser.add_argument("--noise", type=float, default=6e-2,
                        help="noise sigma to inject as input (default to 8e-2)")
    parser.add_argument("--train-sim-time", type=float, default=1000,
                        help="how long to simulate for training set in [s] (default to 1000)")
    parser.add_argument("--val-sim-time", type=float, default=30,
                        help="how long to simulate for validation set in [s] (default to 30)")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="dt for integration in [s] (default to 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for np.random (default to 42)")
    return parser.parse_args()

def generate_data(pendulum, args, validation=False):
    # random initial condition
    pendulum.set_states(theta=np.random.uniform(0, 2 * np.pi), theta_dot=0.)

    if validation:
        steps = int(args.val_sim_time / args.dt)
    else:
        steps = int(args.train_sim_time / args.dt)

    # simulate pendulum
    old_states = pendulum.get_states_array()

    ret = np.zeros((), dtype=[
        ("curr_state", "<f4", (steps, old_states.shape[0])),
        ("next_state", "<f4", (steps, old_states.shape[0])),
        ("control", "<f4", (steps, 2)), # len([tau, dt]) == 2
    ])
    for i in trange(steps):
        # white-noise input
        tau = np.random.randn() * args.noise
        pendulum.step({"tau": tau}, args.dt)
        new_states = pendulum.get_states_array()

        ret["curr_state"][i] = old_states
        ret["next_state"][i] = new_states
        ret["control"][i] = [tau, args.dt]

        old_states = new_states

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
