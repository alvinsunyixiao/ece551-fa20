import argparse
import os
import numpy as np

from tqdm import trange

from models import PlanarRotor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="directory to store the output dataset")
    parser.add_argument("--mass", type=float, default=2, help="mass of the planar rotor")
    parser.add_argument("--length", type=float, default=.5, help="length of the planar roter")
    parser.add_argument("--train-noise", type=float, default=2,
                        help="noise sigma to inject as train data input")
    parser.add_argument("--train-sim-time", type=float, default=100,
                        help="how long to simulate for training set in [s]")
    parser.add_argument("--val-sim-time", type=float, default=30,
                        help="how long to simulate for validation set in [s]")
    parser.add_argument("--val-noise", type=float, default=.2,
                        help="sinusoidal input magnitude for validation data")
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="dt for integration in [s]")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for np.random")
    return parser.parse_args()

def generate_data(rotor, args, validation=False):
    # random initial condition
    rotor.set_states(x=0., y=0., theta=0., x_dot=0., y_dot=0., theta_dot=0.)

    if validation:
        steps = int(args.val_sim_time / args.dt)
    else:
        steps = int(args.train_sim_time / args.dt)

    # simulate planar rotor system
    states = rotor.get_states_array()

    ret = np.zeros((), dtype=[
        ("state", "<f4", (steps, states.shape[0])),
        ("dstate", "<f4", (steps, states.shape[0])),
        ("control", "<f4", (steps, 2)), # two rotors
    ])
    for i in trange(steps):
        if validation:
            ul = rotor.m + np.random.randn() * args.val_noise
            ur = rotor.m + np.random.randn() * args.val_noise
        else:
            ul = rotor.m + np.random.randn() * args.train_noise
            ur = rotor.m + np.random.randn() * args.train_noise

        ul = max(ul, 0)
        ur = max(ur, 0)

        # save current state and input
        ret["state"][i] = states
        ret["control"][i] = [ul, ur]
        dstate_dict = rotor.dynamics({"ul": ul, "ur": ur})
        ret["dstate"][i] = [
            dstate_dict["x"], dstate_dict["y"], dstate_dict["theta"],
            dstate_dict["x_dot"], dstate_dict["y_dot"], dstate_dict["theta_dot"],
        ]

        # step through simulation
        rotor.step({"ul": ul, "ur": ur}, args.dt)
        states = rotor.get_states_array()

    return ret

if __name__ == "__main__":
    args = parse_args()

    # set random seed
    np.random.seed(args.seed)

    # generate dataset
    rotor = PlanarRotor(args.mass, args.length)
    train_dataset = generate_data(rotor, args, False)
    val_dataset = generate_data(rotor, args, True)

    # save dataset to disk
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "train"), train_dataset)
    np.save(os.path.join(args.output_dir, "val"), val_dataset)
