import numpy as np
import gymnasium as gym


class VanillaBitInversionEnv(gym.Env):
    def __init__(self, n_bits=4):
        self.bits = [0] * n_bits
        self.action_space = gym.spaces.Discrete(n_bits + 1)
        self.observation_space = gym.spaces.MultiBinary(n_bits)

    def reset(self, start_state=None):
        if start_state is None:
            self.bits = np.random.randint(0, 2, len(self.bits))
        else:
            assert len(start_state) == len(self.bits)
            self.bits = start_state
        return self.bits

    def step(self, action):
        if action != len(self.bits):
            self.bits[action] = 1 - self.bits[action]
        # last action is a no-op
        return self.bits, sum(self.bits), False, False, {}

    def render(self):
        print(self.bits)


if __name__ == "__main__":
    env = VanillaBitInversionEnv()
    env.reset()