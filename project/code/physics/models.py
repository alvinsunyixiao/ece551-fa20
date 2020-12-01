import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict


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
    def __init__(self, length, mass, damping):
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
        self.ax.grid()
        self.line, = self.ax.plot([], [], 'o-', lw=2)

        return self.line,

    def draw(self):
        theta = self._states["theta"] - np.pi / 2
        self.line.set_data([0, np.cos(theta)], [0, np.sin(theta)])

        return self.line,
