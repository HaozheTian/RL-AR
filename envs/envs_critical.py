import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import matplotlib.pyplot as plt
import math
import do_mpc
import casadi as ca
from collections import deque
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from typing import Dict, Optional, Union


def reset_paras(model_paras: Dict, altered_paras: Dict) -> Dict:
    for key, val in altered_paras.items():
        print(f'{key}:   Model = {model_paras[key]:.2f}  |  Plant = {val:.2f}')
        model_paras[key] = val
    return model_paras


class GlucoseHistory():
    def __init__(self) -> None:
        self.obs_queue = deque(maxlen=2)
    
    def reset(self, glucose):
        self.obs_queue.append(100.0)
        self.obs_queue.append(glucose)
        self.time = 0

    def add_his(self, glucose):
        self.obs_queue.append(glucose)
        self.time += 1

    def get_obs(self):
        return np.array([self.obs_queue[1], 
                         self.obs_queue[1]-self.obs_queue[0], 
                         self.time])


class Glucose(gym.Env):
    """
    Bergman minimal model for blood glucose simulation.

    Based on the parameters in https://www.nature.com/articles/s41598-022-16535-2
    """
    def __init__(self, altered_paras: Dict={}, render_mode: Optional[str]=None, max_steps: int=100, action_penalty: float=2.0):
        self.action_penalty = action_penalty
        self.render_mode = render_mode
        self.max_steps = max_steps
        # model parameters
        model_paras = {
            'Gb': 138,     'Ib': 7,     'n': 0.2814, 'p1': 0,
            'p2': 0.0142,  'p3': 15e-6, 'D0':4,    'dt': 10
        }
        # custom parameters
        for key, val in reset_paras(model_paras, altered_paras).items():
            exec(f'self.{key}={val}')
        # Gym style observation space and action space
        self.observation_space = Box(
            low=np.array([0, -20, 0]),
            high=np.array([1000, 20, 200]),
            dtype=np.float32 # Data type of the observation space
        )
        self.action_space = Box(low=0, high=2, dtype=np.float32)
        self.obs_history = GlucoseHistory()

        Gb, Ib, n, p1, p2, p3, D0, dt = tuple(model_paras.values())
        model = do_mpc.model.Model(model_type='continuous')
        # set model states
        G = model.set_variable(var_type='_x', var_name='G')
        X = model.set_variable(var_type='_x', var_name='X')
        I = model.set_variable(var_type='_x', var_name='I')
        u = model.set_variable(var_type='_u', var_name='u')
        Dt = model.set_variable(var_type='_tvp', var_name='Dt')
        # set rhs
        dG = -p1 * (G - Gb) - G * X + Dt
        dX = -p2 * X + p3 * (I - Ib)
        dI = -n * (I - Ib) + u
        model.set_rhs('G', dG)
        model.set_rhs('X', dX)
        model.set_rhs('I', dI)
        model.setup()

        self.init_state = np.array([self.Gb, 0., self.Ib], dtype=np.float32)

        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=dt)
        tvp_template_sim = simulator.get_tvp_template()
        def tvp_fun_sim(t_now):
            t = t_now
            tvp_template_sim['Dt'] = D0*math.exp(-0.01*t)
            return tvp_template_sim
        simulator.set_tvp_fun(tvp_fun=tvp_fun_sim)
        simulator.setup()
        self.simulator = simulator

    def reset(self, seed: Optional[int]=0) -> Union[np.ndarray, Dict]:
        super().reset(seed=seed)
        # initial states
        self.time_step = 0
        self.simulator.reset_history()
        self.simulator.x0 = np.copy(self.init_state).reshape(-1, 1)
        self.state = np.copy(self.init_state)

        glucose = self.init_state[0]
        self.obs_history.reset(glucose)
        obs = self.obs_history.get_obs()

        self.traj_obs = [glucose]
        self.traj_act = []
        return obs, {"state": self.state, "meas":  np.array([glucose])}

    def step(self, action: np.ndarray) -> Union[np.ndarray, float, bool, bool, Dict]:
        self.time_step += 1
        u = action.reshape(-1,1)
        state_next = self.simulator.make_step(u)
        self.state = state_next.reshape((-1,))
        G = self.state[0]

        term = (G<10) or (G>1000)
        trun = True if self.time_step >= self.max_steps else False
        # Magni risk function https://proceedings.mlr.press/v126/fox20a/fox20a.pdf
        if term:
            reward = -1e5
        else:
            reward = -1 * (3.35506 * (math.log(G)**0.8353 - 3.7932))**2
        
        if action[0] >= 0.1:
            reward -= self.action_penalty

        self.obs_history.add_his(G)
        obs = self.obs_history.get_obs()

        self.traj_obs.append(obs)
        self.traj_act.append(u)
        if self.render_mode=='human' and (term or trun):
            self.render()
        return obs, reward, term, trun, {"state": self.state, "meas":  np.array([G])}
    
    def render(self):
        traj_obs = self.get_plot_obs(np.array(self.traj_obs))
        traj_act = np.array(self.traj_act)

        num_steps = len(traj_obs)
        t = np.arange(0, self.dt*(num_steps-1)+0.001, self.dt)
        t_c = np.arange(0, self.dt*(num_steps-2)+0.001, self.dt/100.0)

        traj_act = traj_act[(t_c//self.dt).astype(int)]
        plt.subplot(2,1,1)
        plt.plot(t, traj_obs)
        plt.subplot(2,1,2)
        plt.plot(t_c, traj_act)
        plt.show()

    def get_plot_obs(self, eps_obs: np.ndarray):
        assert eps_obs.ndim > 1, "_get_plot_obs() deals with obs with shape (N, num_obs)"
        return eps_obs[:,0]


class BiGlucose(gym.Env):
    """
    Extended Bi-glucose Horvorka model for blood glucose simulation.

    Based on the parameters in https://ieeexplore.ieee.org/abstract/document/10252997
    """
    def __init__(self, altered_paras: Dict={}, render_mode: Optional[str]=None, max_steps: int=200, action_penalty: int=3):
        self.action_penalty = action_penalty    # penalty for large action
        self.render_mode = render_mode
        self.max_steps = max_steps
        model_paras = {
            "D_G": 80, "V_G": 0.14, "k_12": 0.0968, "F_01": 0.0119, "EGP_0": 0.0213, 
            "A_g": 0.8, "t_maxG": 40, "t_maxI": 55, "V_I": 0.12, "k_e": 0.138, 
            "k_a1": 0.0088, "k_a2": 0.0302, "k_a3": 0.0118, "k_b1": 7.5768e-05, "k_b2": 1.4194e-05, 
            "k_b3": 0.00085, "t_maxN": 20.59, "k_N": 0.735, "V_N": 23.46, "p": 0.074, 
            "S_N": 19800.0, "M_g": 180.16, "BW": 68.5, "N_b": 48.13, "dt":10}
        # custom parameters
        for key, val in reset_paras(model_paras, altered_paras).items():
            exec(f'self.{key}={np.float32(val)}')
        # Gym style observation space and action space
        self.observation_space = Box(
            low=np.array([0, -10, 0]),
            high=np.array([1000, 10, 200]),
            dtype=np.float32 # Data type of the observation space
        )
        self.action_space = Box(
            low=np.array([0.0, 0.0]), # Lower bounds for each dimension
            high=np.array([1.0, 5]), # Upper bounds for each dimension
            dtype=np.float32 # Data type of the observation space
        )
        self.obs_history = GlucoseHistory()

        self.init_state, self.u_basal = self._solve_steady_state()

        D_G, V_G, k_12, F_01, EGP_0, A_g, t_maxG, t_maxI, V_I, k_e, \
        k_a1, k_a2, k_a3, k_b1, k_b2, k_b3, t_maxN, k_N, V_N, p, S_N, \
        M_g, BW, N_b, dt = tuple(model_paras.values())
        model = do_mpc.model.Model(model_type='continuous')
        # set model states
        Q_1 = model.set_variable(var_type='_x', var_name='Q_1')
        Q_2 = model.set_variable(var_type='_x', var_name='Q_2')
        x_1 = model.set_variable(var_type='_x', var_name='x_1')
        x_2 = model.set_variable(var_type='_x', var_name='x_2')
        x_3 = model.set_variable(var_type='_x', var_name='x_3')
        S_1 = model.set_variable(var_type='_x', var_name='S_1')
        S_2 = model.set_variable(var_type='_x', var_name='S_2')
        I = model.set_variable(var_type='_x', var_name='I')
        Z_1 = model.set_variable(var_type='_x', var_name='Z_1')
        Z_2 = model.set_variable(var_type='_x', var_name='Z_2')
        N = model.set_variable(var_type='_x', var_name='N')
        Y = model.set_variable(var_type='_x', var_name='Y')
        # set inputs
        u_I = model.set_variable(var_type='_u', var_name='u_I')
        u_N = model.set_variable(var_type='_u', var_name='u_N')
        # Set disturbance
        U_G = model.set_variable(var_type='_tvp', var_name='U_G')
        # RHS
        G = Q_1/V_G
        F_01c = ca.if_else(G>=4.5, F_01, F_01*G/4.5)
        F_R = ca.if_else(G>=9, 0.003*(G-9)*V_G, 0)
        dQ_1  = -F_01c - x_1*Q_1 + k_12*Q_2 - F_R + EGP_0*(1-x_3) + 1e3/(M_g*BW)*U_G + Y*Q_1
        dQ_2  = x_1*Q_1 - (k_12 + x_2)*Q_2
        dx_1  = -k_a1*x_1 + k_b1*I
        dx_2  = -k_a2*x_2 + k_b2*I
        dx_3  = -k_a3*x_3 + k_b3*I
        dS_1  = (u_I+self.u_basal[0]) - S_1/t_maxI
        dS_2  = S_1/t_maxI - S_2/t_maxI
        dI  = S_2/(V_I*t_maxI) - k_e*I
        dZ_1  = (u_N*1e-6+self.u_basal[1]) - Z_1/t_maxN
        dZ_2  = Z_1/t_maxN - Z_2/t_maxN
        dN  = -k_N*(N-N_b) + Z_2/(V_N*t_maxN)
        dY  = -p*Y + p*S_N*(N-N_b)
        model.set_rhs('Q_1', dQ_1)
        model.set_rhs('Q_2', dQ_2)
        model.set_rhs('x_1', dx_1)
        model.set_rhs('x_2', dx_2)
        model.set_rhs('x_3', dx_3)
        model.set_rhs('S_1', dS_1)
        model.set_rhs('S_2', dS_2)
        model.set_rhs('I', dI)
        model.set_rhs('Z_1', dZ_1)
        model.set_rhs('Z_2', dZ_2)
        model.set_rhs('N', dN)
        model.set_rhs('Y', dY)
        model.setup()
        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=dt)
        tvp_template_sim = simulator.get_tvp_template()
        def tvp_fun_sim(t_now):
            t = t_now
            tvp_template_sim['U_G'] = D_G*A_g/(t_maxG**2)*t*ca.exp(-t/t_maxG)
            return tvp_template_sim
        simulator.set_tvp_fun(tvp_fun=tvp_fun_sim)
        simulator.setup()
        self.simulator = simulator
    
    def reset(self, seed: Optional[int]=0) -> Union[np.ndarray, Dict]:
        super().reset(seed=seed)
        # initial states
        self.time_step = 0
        self.simulator.reset_history()
        self.simulator.x0 = np.copy(self.init_state).reshape(-1, 1)
        self.state = np.copy(self.init_state)

        glucose = self.init_state[0]/self.V_G*18
        self.obs_history.reset(glucose)
        obs = self.obs_history.get_obs()

        self.traj_obs = [glucose]
        self.traj_act = []
        return obs, {"state": self.state, "meas":  np.array([glucose])}

    def step(self, action: np.ndarray) -> Union[np.ndarray, float, bool, bool, Dict]:
        self.time_step += 1
        u = action.reshape(-1,1)
        state_next = self.simulator.make_step(u)
        self.state = state_next.reshape((-1,))
        G = self.state[0]/self.V_G*18

        term = (G<10) or (G>1000)
        # Magni risk function https://proceedings.mlr.press/v126/fox20a/fox20a.pdf
        if term:
            reward = -1e5
        else:
            reward = -10 * (3.35506 * (math.log(G)**0.8353 - 3.7932))**2
        if action[0] >= 0.1:
            reward -= self.action_penalty
        trun = True if self.time_step >= self.max_steps else False

        self.obs_history.add_his(G)
        obs = self.obs_history.get_obs()
        self.traj_obs.append(G)
        self.traj_act.append(action)

        if self.render_mode=='human' and (term or trun):
            self.render()
        return obs, reward, term, trun, {"state": self.state, "meas": np.array([G])}

    def render(self):
        traj_obs = self.get_plot_obs(np.array(self.traj_obs))
        traj_act = np.array(self.traj_act)

        num_steps = len(traj_obs)
        t = np.arange(0, self.dt*(num_steps-1)+0.001, self.dt)
        t_c = np.arange(0, self.dt*(num_steps-2)+0.001, self.dt/100.0)

        plt.subplot(2,1,1)
        plt.plot(t, traj_obs)
        ax = plt.subplot(2,1,2)
        for i in range(traj_act.shape[1]):
            traj_act_i = traj_act[:, i]
            traj_act_i = traj_act_i[(t_c//self.dt).astype(int)]
            if i == 1:
                ax = ax.twinx()
            plt.plot(t_c, traj_act_i, f'C{i}')
        plt.show()

    def get_plot_obs(self, eps_obs: np.ndarray):
        assert eps_obs.ndim > 1, "_get_plot_obs() deals with obs with shape (N, num_obs)"
        return eps_obs[:,0]

    def _solve_steady_state(self):
        # a + b*k + c*k/(d+e*k) = 0
        a = self.EGP_0 - self.F_01
        b = -((self.EGP_0*self.k_b3)/(self.k_a3*self.k_e*self.V_I) + \
              (7.7*self.V_G*self.k_b1)/(self.k_a1*self.k_e*self.V_I))
        c = 7.7*self.V_G*self.k_b1*self.k_a2*self.k_12
        d = self.k_a1*self.k_a2*self.k_12*self.k_e*self.V_I
        e = self.k_b2*self.k_a1
        # Ak^2 + Bk + C = 0
        A = b*e
        B = (a*e + b*d + c)
        C = a*d
        k1, k2 = (-B+np.sqrt(B**2-4*A*C))/(2*A), (-B-np.sqrt(B**2 - 4*A*C))/(2*A)
        k = max(k1, k2)

        if k < 0:
            raise ValueError("Initial state unsolvable for the BiGlucose parameters")
            
        Q_1 = 7.7*self.V_G
        Q_2 = (7.7*self.V_G*self.k_b1*self.k_a2*k)/  \
            (self.k_a1*self.k_a2*self.k_12*self.k_e*self.V_I + self.k_b2*self.k_a1*k)
        x_1 = self.k_b1 / (self.k_a1*self.V_I*self.k_e) * k
        x_2 = self.k_b2 / (self.k_a2*self.V_I*self.k_e) * k
        x_3 = self.k_b3 / (self.k_a3*self.V_I*self.k_e) * k
        S_1 = self.t_maxI * k
        S_2 = self.t_maxI * k
        I = k/(self.V_I*self.k_e)

        x_steady_state = np.array([Q_1, Q_2, x_1, x_2, x_3, S_1, S_2, I, 0., 0., self.N_b, 0.],
                                  dtype=np.float32)
        u_steady_state = np.array([k, 0.],
                                  dtype=np.float32)
        
        return x_steady_state, u_steady_state


class CSTR(gym.Env):
    """https://www.do-mpc.com/en/latest/example_gallery/CSTR.html"""
    def __init__(self, altered_paras: Dict={}, render_mode: Optional[str]=None, max_steps: int=300):
        self.render_mode = render_mode
        self.max_steps = max_steps
        # model parameters
        model_paras = {
            'alpha': 1.0, 'beta': 1.0,
            'K0_ab': 1.287e12, 'K0_bc': 1.287e12, 'K0_ad': 9.043e9, 'R_gas': 8.3144621e-3,
            'E_A_ab': 9758.3*1.00, 'E_A_bc': 9758.3*1.00, 'E_A_ad': 8560.0*1.0, 'H_R_ab': 4.2,
            'H_R_bc': -11.0, 'H_R_ad': -41.85, 'Rou': 0.9342, 'Cp': 3.01,
            'Cp_k': 2.0, 'A_R': 0.215, 'V_R': 10.01, 'm_k': 5.0,
            'T_in': 130.0, 'K_w': 4032.0, 'C_A0': (5.7+4.5)/2.0*1.0, 'dt': 0.005
        }
        # custom parameters
        for key, val in reset_paras(model_paras, altered_paras).items():
            exec(f'self.{key}={val}')
        # Gym style observation space and action space
        self.observation_space = Box(
            low=np.array([0.1, 0.1, 50, 50]), # Lower bounds for each dimension
            high=np.array([2, 2, 200, 150]), # Upper bounds for each dimension
            dtype=np.float32 # Data type of the observation space
        )
        self.action_space = Box(
            low=np.array([0.5, -8.5]), # Lower bounds for each dimension
            high=np.array([10, 0.0]), # Upper bounds for each dimension
            dtype=np.float32 # Data type of the observation space
        )
        self.init_state = np.array([0.8, 0.5, 134.14, 130.0], dtype=np.float32)

        alpha, beta, K0_ab, K0_bc, K0_ad, R_gas, E_A_ab, E_A_bc, E_A_ad, H_R_ab, \
        H_R_bc, H_R_ad, Rou, Cp, Cp_k, A_R, V_R, m_k, T_in, K_w, \
        C_A0, dt = tuple(model_paras.values())
        model = do_mpc.model.Model(model_type='continuous')
        # set model states
        C_a = model.set_variable(var_type='_x', var_name='C_a')
        C_b = model.set_variable(var_type='_x', var_name='C_b')
        T_R = model.set_variable(var_type='_x', var_name='T_R')
        T_K = model.set_variable(var_type='_x', var_name='T_K')
        F = model.set_variable(var_type='_u', var_name='F')
        Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')
        # set rhs
        K_1 = beta * K0_ab * ca.exp((-E_A_ab)/(T_R+273.15))
        K_2 = K0_bc * ca.exp((-E_A_bc)/(T_R+273.15))
        K_3 = K0_ad * ca.exp((-alpha*E_A_ad)/(T_R+273.15))
        T_dif = T_R - T_K
        F_r = F*10
        Q_dot_r = Q_dot*1000
        model.set_rhs('C_a', F_r*(C_A0 - C_a) -K_1*C_a - K_3*(C_a**2))
        model.set_rhs('C_b', -F_r*C_b + K_1*C_a - K_2*C_b)
        model.set_rhs('T_R', ((K_1*C_a*H_R_ab + K_2*C_b*H_R_bc + K_3*(C_a**2)*H_R_ad)/(-Rou*Cp)) + F_r*(T_in-T_R) +(((K_w*A_R)*(-T_dif))/(Rou*Cp*V_R)))
        model.set_rhs('T_K', (Q_dot_r + K_w*A_R*(T_dif))/(m_k*Cp_k))
        model.setup()

        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=dt)
        simulator.setup()
        self.simulator = simulator

    def reset(self, seed: Optional[int]=0) -> Union[np.ndarray, Dict]:
        super().reset(seed=seed)
        # initial states
        self.time_step = 0
        self.simulator.reset_history()
        self.simulator.x0 = np.copy(self.init_state).reshape(-1, 1)
        self.state = np.copy(self.init_state)

        obs = np.copy(self.state)
        self.traj_obs = [obs]
        self.traj_act = []
        return obs, {"state": self.state, "meas": self.state}

    def step(self, action: np.ndarray) -> Union[np.ndarray, float, bool, bool, Dict]:
        self.time_step += 1
        u = action.reshape(-1,1)
        state_next = self.simulator.make_step(u)
        self.state = state_next.reshape((-1,))
        C_a, C_b, T_R, T_K = self.state[0], self.state[1], self.state[2], self.state[3]

        term = (C_a<0.1 or C_a>2) or (C_a<0.1 or C_a>2) \
               or (T_R<50 or T_R>200) or (T_K<50 or T_K>150)
        trun = True if self.time_step>=self.max_steps else False
        reward = -(100*(C_b - 0.6))**2
        if term:
            reward -= 1e4

        obs = np.copy(self.state)
        self.traj_obs.append(obs)
        self.traj_act.append(action)
        if self.render_mode=='human' and (term or trun):
            self.render()
        return obs, reward, term, trun, {"state": self.state, "meas": self.state}
    
    def render(self):
        traj_obs = self.get_plot_obs(np.array(self.traj_obs))
        traj_act = np.array(self.traj_act)

        num_steps = len(traj_obs)
        t = np.arange(0, self.dt*(num_steps-1)+0.0001, self.dt)
        t_c = np.arange(0, self.dt*(num_steps-2)+0.0001, self.dt/100.0)

        plt.subplot(2,1,1)
        plt.plot(t, traj_obs)
        ax = plt.subplot(2,1,2)
        for i in range(traj_act.shape[1]):
            traj_act_i = traj_act[:, i]
            traj_act_i = traj_act_i[(t_c//self.dt).astype(int)]
            if i == 1:
                ax = ax.twinx()
            plt.plot(t_c, traj_act_i, f'C{i}')
        plt.show()

    def get_plot_obs(self, eps_obs: np.ndarray):
        assert eps_obs.ndim > 1, "_get_plot_obs() deals with obs with shape (N, num_obs)"
        return eps_obs
    
    def plot_traj(self, observations, *actions):
        num_steps = len(observations)
        # plot observations
        t = np.arange(0, self.dt*(num_steps-1)+0.0001, self.dt)
        t_c = np.arange(0, self.dt*(num_steps-2)+0.0001, self.dt/100.0)
        ax0 = plt.subplot(2,2,(1,2))
        plt.plot(t, observations[:, 0], color='C0', label=f'C_a')
        plt.plot(t, observations[:, 1], color='C1', label=f'C_b')
        plt.legend(loc="upper left")
        ax1 = ax0.twinx()
        plt.plot(t, observations[:, 2], color='C2', label=f'T_R')
        plt.plot(t, observations[:, 3], color='C3', label=f'T_K')
        plt.legend(loc="upper right")

        def plot_act(ax, act):
            act_0 = act[(t_c//self.dt).astype(int),0]
            plt.plot(t_c, act_0, f'C4', label="Flow")
            plt.legend(loc="upper left")
            ax = ax.twinx()
            act_1 = act[(t_c//self.dt).astype(int),1]
            plt.plot(t_c, act_1, f'C5', label="Q")
            plt.legend(loc="upper right")

        if len(actions) == 1:
            ax = plt.subplot(2,2,(3, 4))
            plot_act(ax, actions[0])
        else:
            ax = plt.subplot(2,2,3)
            plot_act(ax, actions[0])
            ax = plt.subplot(2,2,4)
            plot_act(ax, actions[1])
        plt.show()
    

class CartPole(CartPoleEnv):
    def __init__(self, altered_paras={}, render_mode = None, max_steps=250, xp=1):
        super().__init__(render_mode)
        self.xp = xp
        self.max_steps = max_steps
        model_paras = {
            'gravity': 9.8, 'masscart': 1.0,  'masspole': 0.1,
            'length': 0.5,  'force_mag': 10., 'dt': 0.02
        }
        for key, val in reset_paras(model_paras, altered_paras).items():
            exec(f'self.{key}={val}')
        self.masstotal = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        obs_high = np.array([4.8, 3.1, 24*math.pi/360, 5.0], dtype=np.float32,)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1., high=1., dtype=np.float32)
        self.init_state = np.array([0., 0., 6*2*math.pi/360, 0], dtype=np.float32)

    def step(self, action):
        self.time_step += 1
        x, x_dot, theta, theta_dot = self.state

        reward = -1000*theta**2 - self.xp*max(0, abs(x)-0.25)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (action.item()*self.force_mag + self.polemass_length*theta_dot**2*sintheta) / self.masstotal
        thetaacc = (self.gravity*sintheta - costheta*temp) / (self.length*(4.0/3.0 - self.masspole*costheta**2/self.masstotal))
        xacc = temp - self.polemass_length*thetaacc*costheta / self.masstotal
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = bool(
            x < -4.8
            or x > 4.8
            or theta < -24*math.pi/360
            or theta > 24*math.pi/360
        )
        if terminated:
            reward -= 1e4
        truncated = False if self.time_step<self.max_steps else True
        if self.render_mode == "human":
            self.render()
        return np.copy(self.state), reward, terminated, truncated, {"state": self.state, "meas": self.state}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.time_step=0
        self.state = np.copy(self.init_state)

        if self.render_mode == "human":
            self.render()
        return np.copy(self.state), {"state": self.state, "meas": self.state}
    
    def get_plot_obs(self, eps_obs: np.ndarray):
        assert eps_obs.ndim > 1, "_get_plot_obs() deals with obs with shape (N, num_obs)"
        eps_obs[:,2] = eps_obs[:,2]/np.pi*180   
        return eps_obs[:,[0,2]]
    
    def plot_traj(self, observations, *actions):
        num_steps = len(observations)
        # plot observations
        t = np.arange(0, self.dt*(num_steps-1)+0.0001, self.dt)
        t_c = np.arange(0, self.dt*(num_steps-2)+0.0001, self.dt/100.0)
        ax0 = plt.subplot(2,2,(1,2))
        plt.plot(t, observations[:, 0], color='C0', label=f'x')
        plt.legend(loc="upper left")
        ax1 = ax0.twinx()
        plt.plot(t, observations[:, 2]/math.pi*180, color='C1', label=r'\theta')
        plt.legend(loc="upper right")

        def plot_act(ax, act):
            act_0 = act[(t_c//self.dt).astype(int),0]
            plt.plot(t_c, act_0, f'C2', label="Force")
            plt.legend(loc="upper left")

        if len(actions) == 1:
            ax = plt.subplot(2,2,(3, 4))
            plot_act(ax, actions[0])
        else:
            ax = plt.subplot(2,2,3)
            plot_act(ax, actions[0])
            ax = plt.subplot(2,2,4)
            plot_act(ax, actions[1])
        plt.show()