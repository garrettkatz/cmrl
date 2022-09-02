# play back precomputed actions
class FixedPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.reset()
    def reset(self):
        self.t = -1
    def __call__(self, observation):
        self.t += 1
        return (self.actions[self.t], None)

