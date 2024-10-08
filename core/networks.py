import torch.nn.functional as F
import torch.nn as nn
import torch

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.action_space.shape[0]+env.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

        self.register_buffer(
            "obs_scale", torch.tensor((env.observation_space.high - env.observation_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "obs_bias", torch.tensor((env.observation_space.high + env.observation_space.low) / 2.0, dtype=torch.float32)
        )
    
    def forward(self, obs, a):
        obs = (obs - self.obs_bias) / self.obs_scale
        a = (a - self.action_bias)/self.action_scale
        x = torch.cat([obs, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Actor(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, env.action_space.shape[0])
        self.fc_logstd = nn.Linear(256, env.action_space.shape[0])

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

        self.register_buffer(
            "obs_scale", torch.tensor((env.observation_space.high - env.observation_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "obs_bias", torch.tensor((env.observation_space.high + env.observation_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN+(LOG_STD_MAX-LOG_STD_MIN)*(log_std+1)*0.5

        return mean, log_std
    
    def get_action(self, obs):
        obs = (obs - self.obs_bias) / self.obs_scale
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # for reparameterization trick (mean + std * N(0,1))
        # this allows backprob with respect to mean and std
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



class BetaNet(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.shape[0]),
        )

        self.network[-1].weight.data.mul_(0.1)
        self.network[-1].bias.data.mul_(0.0)

        self.register_buffer(
            "obs_scale", torch.tensor((env.observation_space.high - env.observation_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "obs_bias", torch.tensor((env.observation_space.high + env.observation_space.low) / 2.0, dtype=torch.float32)
        )
    
    def forward(self, obs):
        obs = (obs - self.obs_bias) / self.obs_scale
        y = self.network(obs)
        y = torch.tanh(0.5*y) * 0.5 + 0.5
        return y


# class BetaNet(nn.Module):
#     def __init__(self, env) -> None:
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(env.observation_space.shape[0], 128),
#             nn.ReLU(),
#             nn.Linear(128, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#         )

#         self.network[-1].weight.data.mul_(0.1)
#         self.network[-1].bias.data.mul_(0.0)

#         self.register_buffer(
#             "obs_scale", torch.tensor((env.observation_space.high - env.observation_space.low) / 2.0, dtype=torch.float32)
#         )
#         self.register_buffer(
#             "obs_bias", torch.tensor((env.observation_space.high + env.observation_space.low) / 2.0, dtype=torch.float32)
#         )
    
#     def forward(self, obs):
#         obs = (obs - self.obs_bias) / self.obs_scale
#         y = self.network(obs)
#         y = torch.tanh(0.5*y) * 0.5 + 0.5
#         return y


class Beta(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.beta = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float))
    
    def forward(self, obs):
        return torch.clamp(self.beta, min=0.0, max=1.0)