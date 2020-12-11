import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D

class PhysicalModel:
    def __init__(self):
        self._states = OrderedDict()

    def set_states(self, **kwargs):
        self._states.update(**kwargs)

    def get_states(self):
        return self._states.copy()

    def get_states_array(self):
        return np.array(list(self._states.values()))

    def measure(self):
        raise NotImplementedError("Derived class should implement this")

    def dynamics(self, inputs):
        raise NotImplementedError("Derived class should implement this")

    def init_draw(self):
        raise NotImplementedError("Derived class should implement this")

    def draw(self):
        raise NotImplementedError("Derived class should implement this")

    def step(self, inputs, dt):
        dstates = self.dynamics(inputs)
        for key in self._states:
            self._states[key] += dt * dstates[key]


class Pendulum(PhysicalModel):
    def __init__(self, length: float, mass: float, damping: float):
        super(Pendulum, self).__init__()
        self.L = length
        self.M = mass
        self.B = damping
        self.J = self.M * self.L**2  # moment of inertia
        self.g = 9.8 # gravity

    def dynamics(self, inputs):
        assert "theta" in self._states and "theta_dot" in self._states, \
            "States of the pendulum are not initialized"

        theta = self._states["theta"]
        theta_dot = self._states["theta_dot"]
        return {
            "theta": theta_dot,
            "theta_dot": (-self.M*self.g*self.L*np.sin(theta) - self.B*theta_dot + inputs["tau"]) / self.J,
        }

    def measure(self):
        return self._states["theta"]

    def init_draw(self):
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
        self.ax.set_aspect('equal')
        self.line, = self.ax.plot([], [], 'o-', lw=2)

        return self.line,

    def draw(self):
        theta = self._states["theta"] - np.pi / 2
        self.line.set_data([0, np.cos(theta)], [0, np.sin(theta)])

        return self.line,

class Lorenz(PhysicalModel):
    def __init__(self, sigma: float, rho: float, beta: float):
        super(Lorenz, self).__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def dynamics(self, inputs):
        assert "x" in self._states and "y" in self._states and "z" in self._states, \
            "States of the Lorenz system are not initialized"

        x = self._states["x"]
        y = self._states["y"]
        z = self._states["z"]
        return {
            "x": self.sigma * (y - x) + inputs["u"],
            "y": x * (self.rho - z) - y,
            "z": x * y - self.beta * z,
        }

    def init_draw(self):
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-30, 30)
        self.ax.set_ylim(-30, 30)
        self.ax.set_zlim(0, 50)

        self.xs = []
        self.ys = []
        self.zs = []
        self.line, = self.ax.plot(self.xs, self.ys, self.zs)

        return self.line,

    def draw(self):
        self.xs.append(self._states["x"])
        self.ys.append(self._states["y"])
        self.zs.append(self._states["z"])

        self.line.set_data(np.array(self.xs), np.array(self.ys))
        self.line.set_3d_properties(np.array(self.zs))

        return self.line,

class PlanarRotor(PhysicalModel):
    def __init__(self, mass: float, length: float):
        super(PlanarRotor, self).__init__()
        self.m = mass
        self.r = length / 2
        self.I = mass * length**2 / 12 # moment of inertia
        self.g = 9.8 # gravity

    def dynamics(self, inputs):
        assert "x" in self._states and "x_dot" in self._states and \
               "y" in self._states and "y_dot" in self._states and \
               "theta" in self._states and "theta_dot" in self._states, \
            "States of the planar rotor are not intialized"

        ul = inputs["ul"]
        ur = inputs["ur"]
        return {
            "x": self._states["x_dot"],
            "y": self._states["y_dot"],
            "theta": self._states["theta_dot"],
            "x_dot": -(ul + ur) * np.sin(self._states["theta"]) / self.m,
            "y_dot": (ul + ur) * np.cos(self._states["theta"]) / self.m - self.g,
            "theta_dot": (ur - ul) * self.r / self.I,
        }

    def init_draw(self):
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')

        self.line, = self.ax.plot([], [])

        return self.line,

    def draw(self):
        x = self._states["x"]
        y = self._states["y"]
        theta = self._states["theta"]

        xbl = x - self.r * np.cos(theta)
        xbr = x + self.r * np.cos(theta)
        xtl = xbl - .5 * self.r * np.sin(theta)
        xtr = xbr - .5 * self.r * np.sin(theta)

        ybl = y - self.r * np.sin(theta)
        ybr = y + self.r * np.sin(theta)
        ytl = ybl + .5 * self.r * np.cos(theta)
        ytr = ybr + .5 * self.r * np.cos(theta)

        self.line.set_data([xtl, xbl, xbr, xtr], [ytl, ybl, ybr, ytr])

        return self.line,
