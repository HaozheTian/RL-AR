import do_mpc
import casadi as ca
import numpy as np
import math
import random
import torch
import gymnasium as gym
from tqdm import tqdm
from IPython.utils import io
from typing import Dict
import matplotlib.pyplot as plt

from envs import BiGlucose, Glucose, CSTR, CartPole

class MPC():
    """
    Basic wrapper of model predictive control (MPC)

    Based on the implementation in do-mpc [https://www.do-mpc.com/en/latest/index.html]. For a specific 
    environment, specify the '_controller_setup()' method. If the state definition is different from the
    observation definition, specify the '_get_state()' method. To record information that is different 
    from the observations, specify the '_recorded_obs()' method.

    Attributes:
        env     (gym.Env): The target environment.
        seed    (Int)    : Secify the random seed, default value is 0.
        horizon (Int)    : Horizon for the MPC controller    
    """
    def __init__(self, env: gym.Env, seed: int=0, horizon: int=100) -> None:
        self.env = env
        self.seed = seed
        self.horizon = horizon

    def get_u(self, obs: np.ndarray) -> np.ndarray:
        """
        Generate control action

        Input:
            obs  (1 row np.ndarray): Observation

        Return:
            u    (1 row np.ndarray): Action
        """
        self.state = self._get_state(obs)
        with io.capture_output() as captured:
            u = self.controller.make_step(self.state).flatten().astype(self.env.action_space.dtype)
            act = np.clip(u, self.env.action_space.low, self.env.action_space.high)
        if hasattr(self, 'model'):
            _, _, _, _, info = self.model.step(act)
            self.state = info["state"]
        return act
        
    def get_pred(self, obs: np.ndarray) -> bool:
        """
        Predict one step into the future
        """
        pass
        
    def initialize(self, obs: np.ndarray) -> None:
        if not hasattr(self, 'controller'):
            self._seed()
            self.controller = self._controller_setup()
        if hasattr(self, 'model'):
            self.model.reset(self.seed)
            self.state = np.copy(self.model.state)
        init_state = self._get_state(obs)
        self.controller.reset_history()
        self.controller.x0 = init_state.reshape((-1, 1))
        self.controller.set_initial_guess()
        self.state = init_state

    def _controller_setup(self):
        raise Exception("No model specified. Customize by inheriting the MPC class")
    
    def _get_state(self, obs:np.ndarray) -> np.ndarray:
        return obs.reshape((-1, 1))
    
    def _get_obs(self, state:np.ndarray) -> np.ndarray:
        return state.reshape((-1,))
    
    def _seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def validate(self, render_mode=None, num_steps=None) -> Dict:
        if num_steps == None:
            num_steps = self.env.max_steps
        if render_mode == 'human':
            self.env.render_mode = 'human'
        episodic_return, eps_length = 0, 0
        obs, info = self.env.reset(seed = self.seed)
        self.initialize(obs)
        observations, actions = [info["meas"]], []
        for i in tqdm(range(num_steps)):
            act = self.get_u(obs)
            obs_next, rew, terminated, truncated, info = self.env.step(act)
            episodic_return, eps_length = episodic_return + rew, eps_length + 1
            observations.append(info["meas"])
            actions.append(act)

            obs = obs_next
            
            if terminated or truncated:
                print(f'episodic return = {episodic_return/eps_length:.4f}')
                print(f'episodic length = {eps_length}')
                break
        
        if render_mode == 'human':
            self.env.close()
            self.env.render_mode = None
        observations, actions = np.array(observations), np.array(actions)

        self._plot_obs_act(observations, actions)

        return {'observations': observations,
                'actions': actions}
    
    def _plot_obs_act(self, observations, actions):
        if hasattr(self.env, 'plot_traj'):
            self.env.plot_traj(observations, actions)
            return
        
        num_steps = len(observations)
        t = np.arange(0, self.env.dt*(num_steps-1)+0.001, self.env.dt)
        t_c = np.arange(0, self.env.dt*(num_steps-2)+0.001, self.env.dt/100.0)
        plt.subplot(2,1,1)
        plt.plot(t, self.env.get_plot_obs(observations))
        ax = plt.subplot(2,1,2)
        for i in range(actions.shape[1]):
            act_i = actions[:,i]
            act_i = act_i[(t_c//self.env.dt).astype(int)]
            if i == 1:
                ax = ax.twinx()
            plt.plot(t_c, act_i, f'C{i}')
        plt.show()
        

class MPCCartPole(MPC):
    def __init__(self, env: gym.Env=CartPole(), seed: int = 0, horizon: int = 100) -> None:
        super().__init__(env, seed, horizon)
        self.model_paras = {
            'gravity': 9.8, 'masscart': 1.0,  'masspole': 0.1,
            'masstotal': 1.1, 'polemass_length': 0.05,
            'length': 0.5,  'force_mag': 10., 'dt': 0.02
        }
        self.state = np.array([0., 0., 6*2*math.pi/360, 0], dtype=np.float32)

    def _controller_setup(self):
        gravity, masscart, masspole, masstotal, polemass_length, \
        length, force_mag, dt = tuple(self.model_paras.values())
        model = do_mpc.model.Model(model_type='discrete')
        # set model states
        x = model.set_variable(var_type='_x', var_name='x')
        x_dot = model.set_variable(var_type='_x', var_name='x_dot')
        theta = model.set_variable(var_type='_x', var_name='theta')
        theta_dot = model.set_variable(var_type='_x', var_name='theta_dot')
        force = model.set_variable(var_type='_u', var_name='force')
        # set rhs
        costheta = ca.cos(theta)
        sintheta = ca.sin(theta)
        temp = (force*force_mag + polemass_length*theta_dot**2*sintheta) / masstotal
        thetaacc = (gravity*sintheta - costheta*temp) / (length*(4.0/3.0 - masspole*costheta**2/masstotal))
        xacc = temp - polemass_length*thetaacc*costheta / masstotal
        model.set_rhs('x', x + dt * x_dot)
        model.set_rhs('x_dot', x_dot + dt * xacc)
        model.set_rhs('theta', theta + dt * theta_dot)
        model.set_rhs('theta_dot', theta_dot + dt * thetaacc)
        model.setup()

        controller = do_mpc.controller.MPC(model)
        setup_mpc = {'n_horizon': self.horizon,
                    't_step': dt,
                    'store_full_solution': False}
        controller.set_param(**setup_mpc)
        # objective
        mterm = 100*x**2 + 100*theta**2
        lterm = 100*x**2 + 100*theta**2
        controller.set_objective(mterm=mterm, lterm=lterm)
        controller.set_rterm(force=100)
        controller.bounds['lower', '_u', 'force'] = -1.0
        controller.bounds['upper', '_u', 'force'] = 1.0
        controller.bounds['lower', '_x', 'theta'] = -12*2*math.pi/360
        controller.bounds['upper', '_x', 'theta'] = 12*2*math.pi/360
        controller.setup()
        return controller


class MPCGlucose(MPC):
    def __init__(self, env: gym.Env, seed: int = 0, horizon: int = 20) -> None:
        super().__init__(env, seed, horizon)
        self.model_paras = {
            'Gb': 138,     'Ib': 7,     'n': 0.2814, 'p1': 0,
            'p2': 0.0142,  'p3': 15e-6, 'D0':4,    'dt': 10
        }
        self.model = Glucose()
        self.model.reset(seed)
        self.state = np.copy(self.model.state)

    def _controller_setup(self, require_model=False):
        Gb, Ib, n, p1, p2, p3, D0, dt = tuple(self.model_paras.values())
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

        controller = do_mpc.controller.MPC(model)
        setup_mpc = {'n_horizon': self.horizon,
                    't_step': dt,
                    'store_full_solution': False}
        controller.set_param(**setup_mpc)
        # objective
        lterm = 10 * (3.35506 * (ca.log(G)**0.8353 - 3.7932))**2
        mterm = 10 * (3.35506 * (ca.log(G)**0.8353 - 3.7932))**2
        rterm = 50*u**2
        controller.set_objective(mterm=mterm, lterm=lterm)
        controller.set_rterm(rterm=rterm)
        #Constraints
        controller.bounds['lower', '_u', 'u'] = 0
        controller.bounds['upper', '_u', 'u'] = 80
        controller.bounds['lower', '_x', 'G'] = 50
        controller.bounds['upper', '_x', 'G'] = 800

        # set up rhs for time varying parameter
        tvp_template_mpc = controller.get_tvp_template()
        def tvp_fun_mpc(t_now):
            for k in range(self.horizon):
                t = t_now + k*dt
                tvp_template_mpc['_tvp', k, 'Dt'] = D0*math.exp(-0.01*t)
            return tvp_template_mpc
        controller.set_tvp_fun(tvp_fun=tvp_fun_mpc)
        controller.setup()
        return controller

    def _get_state(self, obs:np.ndarray) -> np.ndarray:
        G = obs[0]
        self.state[0] = G
        return self.state
    

class MPCCSTR(MPC):
    def __init__(self, env: gym.Env, seed: int = 0, horizon: int = 20) -> None:
        super().__init__(env, seed, horizon)
        self.model_paras = {
            'alpha': 1.0, 'beta': 1.0,
            'K0_ab': 1.287e12, 'K0_bc': 1.287e12, 'K0_ad': 9.043e9, 'R_gas': 8.3144621e-3,
            'E_A_ab': 9758.3, 'E_A_bc': 9758.3, 'E_A_ad': 8560.0, 'H_R_ab': 4.2,
            'H_R_bc': -11.0, 'H_R_ad': -41.85, 'Rou': 0.9342, 'Cp': 3.01,
            'Cp_k': 2.0, 'A_R': 0.215, 'V_R': 10.01, 'm_k': 5.0,
            'T_in': 130.0, 'K_w': 4032.0, 'C_A0': (5.7+4.5)/2.0*1.0, 'dt': 0.005
        }
        self.state = np.array([0.8, 0.5, 134.14, 130.0], dtype=np.float32)

    def _controller_setup(self, require_model=False):
        alpha, beta, K0_ab, K0_bc, K0_ad, R_gas, E_A_ab, E_A_bc, E_A_ad, H_R_ab, \
        H_R_bc, H_R_ad, Rou, Cp, Cp_k, A_R, V_R, m_k, T_in, K_w, \
        C_A0, dt = tuple(self.model_paras.values())
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

        controller = do_mpc.controller.MPC(model)
        setup_mpc = {'n_horizon': self.horizon,
                    't_step': dt,
                    'store_full_solution': False}
        controller.set_param(**setup_mpc)
        _x = model.x
        # objective
        mterm = (_x['C_b'] - 0.6)**2 # terminal cost
        lterm = (_x['C_b'] - 0.6)**2 # stage cost
        controller.set_objective(mterm=mterm, lterm=lterm)
        controller.set_rterm(F=1e-3, Q_dot = 2.5e-4)
        # constraints
        # lower bounds of the states
        controller.bounds['lower', '_x', 'C_a'] = 0.1
        controller.bounds['lower', '_x', 'C_b'] = 0.1
        controller.bounds['lower', '_x', 'T_R'] = 50
        controller.bounds['lower', '_x', 'T_K'] = 50
        # upper bounds of the states
        controller.bounds['upper', '_x', 'C_a'] = 2
        controller.bounds['upper', '_x', 'C_b'] = 2
        controller.bounds['upper', '_x', 'T_R'] = 200
        controller.bounds['upper', '_x', 'T_K'] = 140
        # lower bounds of the inputs
        controller.bounds['lower', '_u', 'F'] = 0.5
        controller.bounds['lower', '_u', 'Q_dot'] = -8.5
        # upper bounds of the inputs
        controller.bounds['upper', '_u', 'F'] = 10
        controller.bounds['upper', '_u', 'Q_dot'] = 0.0

        controller.setup()
        return controller
    

class MPCBiGlucose(MPC):
    def __init__(self, env: gym.Env, seed: int = 0, horizon: int = 20) -> None:
        super().__init__(env, seed, horizon)
        self.model_paras = {
            "D_G": 80, "V_G": 0.14, "k_12": 0.0968, "F_01": 0.0119, "EGP_0": 0.0213, 
            "A_g": 0.8, "t_maxG": 40, "t_maxI": 55, "V_I": 0.12, "k_e": 0.138, 
            "k_a1": 0.0088, "k_a2": 0.0302, "k_a3": 0.0118, "k_b1": 7.5768e-05, "k_b2": 1.4194e-05, 
            "k_b3": 0.00085, "t_maxN": 20.59, "k_N": 0.735, "V_N": 23.46, "p": 0.074, 
            "S_N": 19800.0, "M_g": 180.16, "BW": 68.5, "N_b": 48.13, "dt":10}
        self.model = BiGlucose()
        self.model.reset(seed)
        self.state = np.copy(self.model.state)
        self.u_basal = np.copy(self.model.u_basal)

    def _controller_setup(self):
        D_G, V_G, k_12, F_01, EGP_0, A_g, t_maxG, t_maxI, V_I, k_e, \
        k_a1, k_a2, k_a3, k_b1, k_b2, k_b3, t_maxN, k_N, V_N, p, S_N, \
        M_g, BW, N_b, dt = tuple(self.model_paras.values())
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

        controller = do_mpc.controller.MPC(model)
        setup_mpc = {'n_horizon': self.horizon,
                    't_step': dt,
                    'store_full_solution': False}
        controller.set_param(**setup_mpc)

        # objectives
        lterm = 10 * (3.35506 * (ca.log(Q_1/V_G*18)**0.8353 - 3.7932))**2
        mterm = 10 * (3.35506 * (ca.log(Q_1/V_G*18)**0.8353 - 3.7932))**2
        rterm = 100*u_I**2 + 1*u_N**2
        controller.set_objective(mterm=mterm, lterm=lterm)
        controller.set_rterm(rterm=rterm)

        # constraints
        controller.bounds['lower', '_u', 'u_I'] = 0
        controller.bounds['upper', '_u', 'u_I'] = 4
        controller.bounds['lower', '_u', 'u_N'] = 0
        controller.bounds['upper', '_u', 'u_N'] = 5
        controller.bounds['lower', '_x', 'Q_1'] = 50/18*V_G
        controller.bounds['upper', '_x', 'Q_1'] = 800/18*V_G

        tvp_template_mpc = controller.get_tvp_template()
        def tvp_fun_mpc(t_now):
            for k in range(self.horizon):
                t = t_now+k*dt
                tvp_template_mpc['_tvp', k, 'U_G'] = D_G*A_g/(t_maxG**2)*t*ca.exp(-t/t_maxG)
            return tvp_template_mpc
        controller.set_tvp_fun(tvp_fun=tvp_fun_mpc)
        controller.setup()
        return controller

    def _get_state(self, obs:np.ndarray) -> np.ndarray:
        Q_1 = obs[0]/18*self.model_paras["V_G"]
        self.state[0] = Q_1
        return self.state