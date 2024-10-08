import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import gymnasium as gym
from typing import Dict, Union, Optional
from tqdm import tqdm
import os
import fnmatch
import matplotlib.pyplot as plt
from datetime import datetime

from core.networks import SoftQNetwork, Actor, BetaNet, Beta
from core.buffers import ReplayBuffer, ModelBuffer
from core.controllers import *
from core.utils import plot_actions

class SAC():
    """
    Soft Actor Critic (SAC) RL agent. Only works for continuous action space.

    Attributes:
        env         (gym.Env)       : The target environment.
        seed        (Int)           : Secify the random seed, default value is 0.
        rb          (ReplayBuffer)  : Replay buffer.
        global_step (Int)           : Total steps taken.
        num_eps     (Int)           : Number of episodes simulated.
        episodic_returns (List)     : A list of cumulative episodic returns.
        eps_lens    (List)          : A list of episode lengths.
    """
    def __init__(self, 
                 env: gym.Env,
                 ckpt_path: str ='',
                 seed: int = 0, 
                 hyperparameters: Dict = {},
                 autosave = False,
                 use_tb=False) -> None:
        """
        Initialization of the environment.

        Inputs:
            env  (gym.Env)       : The target environment.
            seed (Int)           : Random seed used for the env, torch, numpy, and random.
            hyperparameters (Int): Specified hyperparameters.
            use_tb (Bool)        : Whether to use tensor board.
        """
        self.env = env
        self.env_name = env.__class__.__name__
        self.autosave = autosave
        print('-' * 20)
        print(f'SAC on {self.env_name}')
        self.seed = seed
        self.ckpt_path = ckpt_path
        self._seed()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Runing on {self.device}')
        # initialization
        self._init_hyperparameters()
        self._specify_hyperparameters(hyperparameters)
        self._init_networks()
        self._init_alpha()
        self._init_buffer()
        # logging status
        self.global_step = 0
        self.num_eps = 0
        self.num_fails = 0
        self.episodic_returns = []
        self.eps_lengths = []
        self.use_tb = use_tb
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/SAC'+self.env.__class__.__name__+datetime.now().strftime("%m_%d_%Y_%H_%M"))
    

    def learn(self):
        episodic_return, eps_length = 0, 0
        observations, acts = [], []
        obs, info = self.env.reset(self.seed)
        for global_step in tqdm(range(self.total_timesteps)):
            self.global_step = global_step
            # Get action
            act = self.get_act(obs)
            if not self.env.action_space.contains(act):
                print(act)
            assert self.env.action_space.contains(act), "Invalid action in the action space"
            # Step in the environment
            obs_next, rew, term, trun, info = self.env.step(act)
            done = term or trun
                
            self.rb.add(obs, obs_next, act, rew, done)
            # Logging
            self.num_fails += term
            episodic_return, eps_length = episodic_return + rew, eps_length + 1
            observations.append(info["meas"])
            acts.append(act)
            # Important, easy to overlook
            obs = obs_next

            if self.use_tb and self.autotune_alpha:
                self.writer.add_scalar('charts/alpha', self.alpha, global_step)

            if done:
                if term:
                    episodic_return -= rew
                    eps_length -= 1
                self.num_eps += 1
                self.episodic_returns.append(episodic_return)
                self.eps_lengths.append(eps_length)
                observations.append(info["meas"])
                if self.use_tb:
                    self.writer.add_scalar('charts/failures', 
                        self.num_fails, self.num_eps)
                    self.writer.add_scalar('charts/episodic_return', 
                        episodic_return/eps_length, self.num_eps)
                    self.writer.add_scalar('charts/episode_length', 
                        eps_length, self.num_eps)
                if self.show_full_result:
                    self._plot_obs_act(np.array(observations), np.array(acts))
                    self._print_log()

                episodic_return, eps_length = 0, 0
                observations, acts = [], []
                obs, _ = self.env.reset(self.seed)

            if self.autosave and (global_step+1)%5000 == 0:
                self.save_ckpt(save_path=f'sac_ckpt/{self.env_name}_{global_step}_' + datetime.now().strftime("%H_%M_%S") + '.pt')
            
            if global_step >= self.learning_starts:
                self.update()
    

    def update(self):
        data = self.rb.sample(self.batch_size)
        # compute target for the Q functions
        with torch.no_grad():
            act_next, act_next_log_prob, _ = self.actor.get_action(data.obs_next)
            qf1_next = self.qf1_target(data.obs_next, act_next)
            qf2_next = self.qf2_target(data.obs_next, act_next)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha*act_next_log_prob
            y = data.reward.flatten() + (1 - data.done.flatten()) * self.gamma * (min_qf_next).view(-1)
        # compute loss for the  Q functions
        qf_1 = self.qf1(data.obs, data.act).view(-1)
        qf_2 = self.qf2(data.obs, data.act).view(-1)
        qf1_loss = F.mse_loss(qf_1, y)
        qf2_loss = F.mse_loss(qf_2, y)
        qf_loss = qf1_loss + qf2_loss
        # update the Q functions
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # update the policy
        if self.global_step % self.policy_frequency == 0: # TD3 Delayed update support
            for _ in range(self.policy_frequency):
                acts, log_prob, _ = self.actor.get_action(data.obs)
                qf1 = self.qf1(data.obs, acts)
                qf2 = self.qf2(data.obs, acts)
                min_qf = torch.min(qf1, qf2).view(-1)
                actor_loss = ((self.alpha * log_prob) - min_qf).mean()
                # update parameters
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune_alpha:
                    self._update_alpha(data.obs)

        # update the target network
        if self.global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)      

    
    def validate(self, num_steps=None) -> Dict:
        if num_steps == None:
            num_steps = self.env.max_steps
        episodic_return, eps_length = 0, 0
        obs, info = self.env.reset(seed=self.seed)
        observations, actions = [info["meas"]], []
        for i in tqdm(range(num_steps)):
            _, _, act = self.actor.get_action(torch.Tensor(obs).unsqueeze(0).to(self.device))
            act = act.squeeze(0).detach().cpu().numpy()
            obs_next, rew, term, trun, info = self.env.step(act)
            done = term or trun
            episodic_return, eps_length = episodic_return + rew, eps_length + 1
            observations.append(info["meas"])
            actions.append(act)
            # Important, easy to overlook
            obs = obs_next

            if done:
                print(f'episodic return = {episodic_return/eps_length:.4f}')
                print(f'episodic length = {eps_length}')
                break
        
        observations, actions = np.array(observations), np.array(actions)

        self._plot_obs_act(observations, actions)

        return {'observations': np.array(observations),
                'actions': np.array(actions)}


    def get_act(self, obs: np.ndarray) -> np.ndarray:
        if self.ckpt_path == '' and self.global_step<self.learning_starts:
            act = self.env.action_space.sample()
        else:
            act, _, _ = self.actor.get_action(torch.Tensor(obs).unsqueeze(0).to(self.device))
            act = act.squeeze(0).detach().cpu().numpy()
        return act


    def _init_hyperparameters(self):
        self.q_lr = 1e-3                                # learning rate for Q network
        self.policy_lr = 3e-4                           # learning rate for policy network
        self.buffer_size = 1000000                      # replay buffer size
        self.batch_size = 256                           # batch size for updating network
        self.total_timesteps = 200000                   # maximum number of iterations
        self.learning_starts = self.batch_size          # start learning
        self.tau = 0.005                                # for updating Q target
        self.gamma = 0.99                               # forgetting factor
        self.policy_frequency = 2                       # frequency for updating policy network
        self.target_network_frequency = 1               # frequency for updating target network
        self.show_full_result = False                   # frequency for displaying logs
        self.validate_freq = 100                        # freqeuncy for validation
        self.render_val = False                         # Render validation or not
        self.plot_val_traj = True                       # Plot the trajectory during validation
        self.autotune_alpha = False                     # Automatic entropy tuning


    def _specify_hyperparameters(self, hyperparameters: Dict):
        # rewrite default hyperparameters
        for param, val in hyperparameters.items():
            exec(f'self.{param}={val}')

    
    def _seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)

    
    def _init_networks(self):
        self.actor = Actor(self.env).to(self.device)
        self.qf1 = SoftQNetwork(self.env).to(self.device)
        self.qf2 = SoftQNetwork(self.env).to(self.device)
        self.qf1_target = SoftQNetwork(self.env).to(self.device)
        self.qf2_target = SoftQNetwork(self.env).to(self.device)
        if self.ckpt_path == '':
            print('Training from scratch')
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
        else:
            print(f'Training from the checkpoint in {self.ckpt_path}')
            self._load_ckpt(torch.load(self.ckpt_path, weights_only=True))
        self.q_optimizer = Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = Adam(list(self.actor.parameters()), lr=self.policy_lr)
    

    def _init_buffer(self):
        self.rb = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,    
        )


    def save_ckpt(self, save_path: Optional[str]=None):
        if save_path == None:
            directory = "saved"
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = "SAC_" + self.env.__class__.__name__ + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M") + ".pt"
            path = os.path.join(directory, filename)
        else:
            path = save_path
        
        torch.save({"actor_state_dict": self.actor.state_dict(), 
            "qf1_state_dict": self.qf1.state_dict(),
            "qf2_state_dict": self.qf2.state_dict(),
            "qf1_target_state_dict": self.qf1_target.state_dict(),
            "qf2_target_state_dict": self.qf2_target.state_dict()
            }, path)

        print(f"Checkpoint saved to {path}")


    def _load_ckpt(self, ckpt: dict):
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.qf1.load_state_dict(ckpt["qf1_state_dict"])
        self.qf1_target.load_state_dict(ckpt["qf1_target_state_dict"])
        self.qf2.load_state_dict(ckpt["qf2_state_dict"])
        self.qf2_target.load_state_dict(ckpt["qf2_target_state_dict"])
    

    def _print_log(self):
        print(f'-----------------------------------------')
        print(f'Simulated time steps:            |  {self.global_step}')
        print(f'Simulated episodes:              |  {self.num_eps}')
        print(f'-----------------------------------------')
        print(f'Average episodic return:         |  {sum(self.episodic_returns)/len(self.episodic_returns):.2f}')
        print(f'Average episode length:          |  {sum(self.eps_lengths)/len(self.eps_lengths):.2f}')
        print(f'Most recent episodic return:     |  {self.episodic_returns[-1]:.2f}')
        print(f'Most recent episode length:      |  {self.eps_lengths[-1]:.2f}')
        

    def _plot_obs_act(self, observations: np.ndarray, *actions: np.ndarray):
        if hasattr(self.env, 'plot_traj'):
            self.env.plot_traj(observations, *actions)
            return
        num_steps = len(observations)
        # plot observations
        plt.subplot(2,2,(1,2))
        t = np.arange(0, self.env.dt*(num_steps-1)+0.001, self.env.dt)
        plt.plot(t, self.env.get_plot_obs(observations))

        # plot actions
        t_c = np.arange(0, self.env.dt*(num_steps-2)+0.001, self.env.dt/100.0)
        num_set = len(actions)
        if num_set == 1:
            ax = plt.subplot(2,2,(3,4))
            plot_actions(ax, actions[0], t_c, self.env.dt)
        elif num_set == 2:
            ax1 = plt.subplot(2,2,3)
            plot_actions(ax1, actions[0], t_c, self.env.dt)
            ax2 = plt.subplot(2,2,4)
            plot_actions(ax2, actions[1], t_c, self.env.dt)
        else:
            raise Exception('Too many action groups')
        plt.show()

    
    def _init_alpha(self):
        if self.autotune_alpha:
            self.target_entropy = -self.env.action_space.shape[0]
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.q_lr)
        else:
            self.alpha = 0.2
    

    def _update_alpha(self, obs: np.ndarray):
        with torch.no_grad():
            _, log_pi, _ = self.actor.get_action(obs)
        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()



class RLAR(SAC):
    """
    Attention Integrated Model-based RL agent. Only works for continuous action space.

    Attributes:
        env         (gym.Env)       : The target environment.
        state_dependent (bool)      : Whether to use state-dependent \beta
        seed        (int)           : random seed
        hyperparameters (dict)      : altered RLAR hyperparameters
        use_tb      (bool)          : whether to use tensorboard
        sac_pt      (string)        : path if use SAC-PT
        clpt_path   (string)        : path if loading checkpoint
    """
    def __init__(self, 
                 env: gym.Env,
                 state_dependent: bool = True, 
                 seed: int = 0, 
                 hyperparameters: Dict = {}, 
                 use_tb=False,
                 autosave = False,
                 sac_pt = '',
                 ckpt_path='') -> None:
        self.env = env
        self.env_name = env.__class__.__name__
        self.autosave = autosave
        print('-' * 20)
        print(f'RLAR on {self.env_name}')
        self.sac_pt = sac_pt
        self.ckpt_path = ckpt_path
        self.Beta_network = BetaNet if state_dependent else Beta
        self.seed = seed
        self._seed()
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                    else 'cpu')
        print(f'Runing on {self.device}')
        # initialization
        self._init_hyperparameters()
        self._specify_hyperparameters(hyperparameters)
        self._init_mpc()
        self._init_networks()
        self._init_alpha()
        self._init_buffer()
        # logging status
        self.global_step = 0
        self.num_eps = 0
        self.num_fails = 0
        self.episodic_returns = []
        self.eps_lengths = []
        self.use_tb = use_tb
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/RLAR'+self.env.__class__.__name__+'_'+datetime.now().strftime("%m_%d_%Y_%H_%M"))


    def learn(self):
        episodic_return, eps_length = 0, 0
        episodic_beta = np.zeros(self.env.action_space.shape, dtype=np.float32)
        observations, acts, acts_mpc, acts_sac = [], [], [], []
        obs, info = self.env.reset(self.seed)
        self.mpc.initialize(obs)
        for global_step in tqdm(range(self.total_timesteps)):
            self.global_step = global_step
            # Get action
            act, act_mpc, act_sac, beta = self.get_act(obs, eps_length)
            if self.use_tb:
                for i in range(len(beta)):
                    self.writer.add_scalar(f'charts/beta{i}', beta[i], global_step)
            if not self.env.action_space.contains(act):
                print(act)
            assert self.env.action_space.contains(act), f"Invalid action {act} in the action space"
            # Step in the environment
            obs_next, rew, term, trun, info = self.env.step(act)
            done = term or trun
            # Record transition
            self.rb.add(obs, obs_next, act, act_mpc, rew, done)
            # Logging
            self.num_fails += term
            episodic_return = episodic_return + rew
            episodic_beta = episodic_beta + beta
            eps_length += 1
            observations.append(info["meas"])
            acts.append(act)
            acts_mpc.append(act_mpc)
            acts_sac.append(act_sac)
            # Important, easy to overlook
            obs = obs_next

            if done:
                if term:
                    episodic_return -= rew
                    eps_length -= 1
                self.num_eps += 1
                self.episodic_returns.append(episodic_return)
                self.eps_lengths.append(eps_length)
                observations.append(info["meas"])
                
                if self.use_tb:
                    self.writer.add_scalar('charts/failures', 
                                           self.num_fails, self.num_eps)
                    self.writer.add_scalar('charts/episodic_return', 
                                           episodic_return/eps_length, self.num_eps)
                    self.writer.add_scalar('charts/episode_length', 
                                           eps_length, self.num_eps)
                    for i in range(len(episodic_beta)):
                        self.writer.add_scalar(f'charts/episodic_beta{i}', 
                                               episodic_beta[i]/eps_length, self.num_eps)
                if self.show_full_result:
                    self._plot_obs_act(np.array(observations),
                                    np.array(acts), np.array(acts_sac))
                    self._print_log()
                    for i in range(len(episodic_beta)):
                        print(f'Most recent episode beta{i}:        |  {episodic_beta[i]/eps_length:.4f}')

                episodic_return, eps_length = 0, 0
                episodic_beta = np.zeros(self.env.action_space.shape, dtype=np.float32)
                observations, acts, acts_mpc, acts_sac = [], [], [], []
                obs, _ = self.env.reset(self.seed)
                self.mpc.initialize(obs)

            if global_step >= self.learning_starts:
                self.update()
            
            if self.autosave and (global_step+1)%2000 == 0:
                self.save_ckpt(save_path=f'rlar_ckpt/{self.env_name}_{global_step+1}_' + datetime.now().strftime("%H_%M_%S") + '.pt')


    def update(self):
        data = self.rb.sample(self.batch_size)

        # compute target for the Q functions
        with torch.no_grad():
            act_next, act_next_log_prob, _ = self.actor.get_action(data.obs_next)
            qf1_next = self.qf1_target(data.obs_next, act_next)
            qf2_next = self.qf2_target(data.obs_next, act_next)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha*act_next_log_prob
            y = data.reward.flatten() + (1 - data.done.flatten()) * self.gamma * (min_qf_next).view(-1)
        # compute loss for the  Q functions
        qf_1 = self.qf1(data.obs, data.act).view(-1)
        qf_2 = self.qf2(data.obs, data.act).view(-1)
        qf1_loss = F.mse_loss(qf_1, y)
        qf2_loss = F.mse_loss(qf_2, y)
        qf_loss = qf1_loss + qf2_loss
        # update the Q functions
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # update the policy
        if self.global_step % self.policy_frequency == 0: # TD3 Delayed update support
            for _ in range(self.policy_frequency):
                acts, log_prob, _ = self.actor.get_action(data.obs)
                qf1 = self.qf1(data.obs, acts)
                qf2 = self.qf2(data.obs, acts)
                min_qf = torch.min(qf1, qf2).view(-1)
                actor_loss = ((self.alpha * log_prob) - min_qf).mean()
                # update parameters
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune_alpha:
                    self._update_alpha(data.obs)

        # update the target network
        if self.global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)   
        
        # update attention factor beta
        with torch.no_grad():
            act_sac, _, _ = self.actor.get_action(data.obs)
        beta = self.beta(data.obs)
        actions = beta*data.act_model + (torch.ones_like(beta)-beta)*act_sac
        qf1_beta = self.qf1(data.obs, actions)
        qf2_beta = self.qf2(data.obs, actions)
        beta_loss = -1*(torch.min(qf1_beta, qf2_beta).view(-1).mean())

        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        self.beta_optimizer.step()

        if self.beta.__class__.__name__ == 'Beta':
            if self.beta.beta.data > 0.995:
                self.beta.beta.data = torch.tensor(0.995).to(self.device)

    def get_act(self, obs: np.ndarray, eps_length: int)-> Union[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Get the combined action of MPC and RL

        Returns the combined action, \beta(s)*a_MPC, (1-\beta(s))*a_RL, and \beta
        """
        if self.global_step<self.learning_starts:
            # Take MPC actions
            act_mpc = self.mpc.get_u(obs)
            act_sac = np.zeros_like(act_mpc)
            beta = self.beta(torch.Tensor(obs).to(self.device))
            beta = beta.detach().cpu().numpy()
        else:
            # Take RL actions
            beta = self.beta(torch.Tensor(obs).to(self.device))
            beta = beta.detach().cpu().numpy()

            act_mpc = self.mpc.get_u(obs)
            act_sac, _, _ = self.actor.get_action(torch.Tensor(obs).unsqueeze(0).to(self.device))
            act_sac = act_sac.squeeze(0).detach().cpu().numpy()
        
        act = beta * act_mpc + (np.ones_like(beta)-beta) * act_sac
        act = np.clip(act, self.env.action_space.low, self.env.action_space.high)
        return act, act_mpc, act_sac, beta


    def _init_hyperparameters(self):
        super()._init_hyperparameters()
        self.use_pretrained_beta = True             # Use pretrained beta network
        self.loss_thresh = 1e-8                     # Beta pretraining loss threshold
        self.learning_starts = self.batch_size      # start learning
        self.beta_lr = 5e-6                         # learning rate for the beta network
        self.show_full_result = False               # show all trajectories using plots
        self.horizon = 20                           # horizon for MPC


    def _init_networks(self):
        self.actor = Actor(self.env).to(self.device)
        self.qf1 = SoftQNetwork(self.env).to(self.device)
        self.qf2 = SoftQNetwork(self.env).to(self.device)
        self.qf1_target = SoftQNetwork(self.env).to(self.device)
        self.qf2_target = SoftQNetwork(self.env).to(self.device)
        self.beta = self.Beta_network(self.env).to(self.device)

        self.q_optimizer = Adam(list(self.qf1.parameters())+list(self.qf2.parameters()),
                                lr=self.q_lr)
        self.actor_optimizer = Adam(list(self.actor.parameters()),
                                lr=self.policy_lr)
        self.beta_optimizer = Adam(list(self.beta.parameters()),
                                lr=self.beta_lr)

        if self.ckpt_path == '':
            print('Training from scratch')
            if self.sac_pt == '':
                self.qf1_target.load_state_dict(self.qf1.state_dict())
                self.qf2_target.load_state_dict(self.qf2.state_dict())
            else:
                super()._load_ckpt(torch.load(self.sac_pt, weights_only=True))

            # use scalar beta
            if self.beta.__class__.__name__ == 'Beta':
                return
            # use state-dependent beta
            if self.use_pretrained_beta:
                beta_ckpt_pth = fnmatch.filter(os.listdir('saved'),
                                        f"beta_{self.env.__class__.__name__}*.pt")
                if len(beta_ckpt_pth)==1:
                    beta_ckpt_pth = os.path.join('saved',beta_ckpt_pth[0])
                    ckpt = torch.load(beta_ckpt_pth, weights_only=True)
                    self.beta.load_state_dict(ckpt["beta_state_dict"])
                    print(f"Pretrained beta checkpoint loaded from {beta_ckpt_pth}")
                else:
                    print("No pretrained beta checkpoint, call _pre_train_beta() first")
        else:
            print(f'Training from the checkpoint in {self.ckpt_path}')
            self._load_ckpt(torch.load(self.ckpt_path, weights_only=True))


    def _pre_train_beta(self, lr: float=1e-5,
                        batch_rand_obs_size: int=512,
                        max_num_pt: int=200000):
        optimizer = Adam(list(self.beta.parameters()), lr=lr)
        for i in tqdm(range(max_num_pt)):
            low_rep = np.tile(self.env.observation_space.low,
                              (batch_rand_obs_size, 1))
            high_rep = np.tile(self.env.observation_space.high,
                               (batch_rand_obs_size, 1))
            batch_rand_obs = np.random.uniform(low=low_rep, high=high_rep)
            pred = self.beta(torch.tensor(batch_rand_obs,
                                          dtype=torch.float32).to(self.device))
            loss = F.mse_loss(pred, torch.ones_like(pred)*0.999)
            if loss < self.loss_thresh:
                print(f"Pretraining of Beta finished, batch mean={pred.mean():.4f}")
                if not os.path.exists("saved"):
                    os.makedirs("saved")
                beta_ckpt_path = os.path.join("saved",
                    f"beta_{self.env_name}_thresh_{self.loss_thresh:g}.pt")
                torch.save({"beta_state_dict": self.beta.state_dict()}, beta_ckpt_path)
                print(f"Pretrained Beta saved to {beta_ckpt_path}")
                return
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Pretraining of Beta did not satisfy the given threshold")
        print(f"Pretrained Beta was not saved")
        return

    def _init_buffer(self):
        self.rb = ModelBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,    
        )

    def save_ckpt(self, save_path: Optional[str]=None):
        if save_path == None:
            directory = "saved"
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = "RLAR_" + self.env_name + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M") + ".pt"
            path = os.path.join(directory, filename)
        else:
            path = save_path
        torch.save({"actor_state_dict": self.actor.state_dict(), 
            "qf1_state_dict": self.qf1.state_dict(),
            "qf2_state_dict": self.qf2.state_dict(),
            "qf1_target_state_dict": self.qf1_target.state_dict(),
            "qf2_target_state_dict": self.qf2_target.state_dict(),
            "beta_state_dict": self.beta.state_dict(),
            }, path)
        print(f"Checkpoint saved to {path}")
    
    def _load_ckpt(self, ckpt: dict):
        super()._load_ckpt(ckpt)
        self.beta.load_state_dict(ckpt["beta_state_dict"])

    def _init_mpc(self):   
        mpc_class = globals().get(f'MPC{self.env_name}')
        if mpc_class is None:
            raise ValueError(f"No MPC class found for '{self.env_name}'")
        env_class = globals().get(self.env_name)
        if env_class is None:
            raise ValueError(f"No environment class found for '{self.env_name}'")
        self.mpc = mpc_class(env=env_class(), horizon=self.horizon)
        self.env_model = env_class()

    def validate(self, num_steps=None) -> Dict:
        if num_steps == None:
            num_steps = self.env.max_steps
        episodic_return, eps_length = 0, 0
        episodic_beta = np.zeros(self.env.action_space.shape, dtype=np.float32)
        obs, info = self.env.reset(seed=self.seed)
        observations, acts, acts_mpc, acts_sac = [info["meas"]], [], [], []
        self.mpc.initialize(obs)
        for i in tqdm(range(num_steps)):
            beta = self.beta(torch.Tensor(obs).to(self.device))
            beta = beta.detach().cpu().numpy()
            act_mpc = self.mpc.get_u(obs)
            _, _, act_sac = self.actor.get_action(torch.Tensor(obs).unsqueeze(0).to(self.device))
            act_sac = act_sac.squeeze(0).detach().cpu().numpy()
            act = beta * act_mpc + (np.ones_like(beta)-beta) * act_sac
            act = np.clip(act, self.env.action_space.low, self.env.action_space.high)

            obs_next, rew, terminated, truncated, info = self.env.step(act)
            done = terminated or truncated

            episodic_return = episodic_return + rew
            episodic_beta = episodic_beta + beta
            eps_length += 1
            observations.append(info["meas"])
            acts.append(act)
            acts_mpc.append(act_mpc)
            acts_sac.append(act_sac)

            obs = obs_next

            if done:
                print(f'episodic return = {episodic_return/eps_length:.4f}')
                print(f'episodic length = {eps_length}')
                break

        if self.render_val:
            self.env.render_mode = None

        observations, acts_mpc, acts_sac = np.array(observations), np.array(acts_mpc), np.array(acts_sac)

        self._plot_obs_act(np.array(observations),
                        np.array(acts), np.array(acts_sac))

        episodic_return, eps_length = 0, 0
        episodic_beta = np.zeros(self.env.action_space.shape, dtype=np.float32)

        return {'observations': observations,
                'actions': acts,
                'actions_mpc': acts_mpc, 
                'actions_sac': acts_sac}
