import torch as tr

# Play back precomputed actions
class FixedPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.reset()
    def reset(self):
        self.t = -1
    def __call__(self, observation):
        self.t += 1
        return (self.actions[self.t], None)

# For deep policy gradient methods that output means of uncorrelated Normal exploration noise
class NormalPolicy:
    def __init__(self, net, stdev):
        """
        net is the pytorch policy network that outputs an action mean mu
        returned actions are sampled from a Normal centered at mu with standard deviation stdev
        """
        self.net = net
        self.stdev = stdev
        self.reset()

    def reset(self, explore=True):
        """
        set exploration flag to true or false
        use true during training, false during testing
        when false, network output mu is the returned action
        when true, actions are sampled from normal centered at mu
        """
        self.explore = explore

    def __call__(self, observation, action=None):
        """
        calls the policy network on the provided observation
        when collecting rollouts, use action == None to randomly sample actions from the output distribution
        when gradient-updating from previously collected rollouts, provide the actions from the rollouts
        
        returns action and its log probability
        """

        # normal random exploration centered around predicted action mu
        mu = self.net(tr.tensor(observation, dtype=tr.float))
        dist = tr.distributions.Normal(mu, self.stdev)

        # sample random action if not already provided from previous rollout
        if action is None:
            action = dist.sample() if self.explore else mu
        else:
            action = tr.tensor(action, dtype=tr.float)

        # calculate log_prob, summed across action dim
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.numpy(), log_prob



