"""
Model for a beam from Dan's python VEP code.
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.interpolate import interp1d
import collections

class Beam:
    def __init__(self, name='30L', V0=50000, cur0=30, epsN_coeff=[-0.00444, 0.9977], epsT_coeff=[0.0, 0.0, 1.0],
                 qNZ=13, qSrc=11, RL=0.95, dd=0.97, kprobe=2.0, ol=1, Rtan=1.149, tauV=0.1, tauI=0.1,t0=0.0,voltage_rate_limit=30000,
                 optimal_perveances=np.array([1.8,1.9,2.0])):
        self.name = name
        self.V0 = V0
        self.epsN_coeff = epsN_coeff
        self.epsT_coeff = np.array(epsT_coeff) / 100.0
        self.V = V0
        self.t = t0
        self.optimal_perveance = interp1d(np.array([50000,65000,80000]),optimal_perveances,fill_value='extrapolate') # at 50kV, 65kV, and 80kV
        self.cur0 = self.cur_from_perveance(self.V0, self.optimal_perveance(self.V0), micropervs=True)
        self.cur = self.cur0
        self.kprobe = kprobe
        self.qNZ = qNZ
        self.qSrc = qSrc
        self.RL = RL
        self.dd = dd
        self.ol = ol
        self.Rtan = Rtan
        self.tauV = tauV
        self.tauI = tauI
        self.voltage_rate_limit = voltage_rate_limit
        self.mu = 2
        self.mp = 1.6726 * 10 ** (-27)
        self.e = 1.6022 * 10 ** (-19)
        self.etaT = self.Rtan * np.sqrt(2 * self.mp * self.mu / self.e)
        self.fc_coef = np.array([[-0.109171, 0.0144685, -7.83224 * 10.0 ** -5.0],
                                 [0.0841037, 0.0025516, -7.42683 * 10.0 ** -8.0]])
        self.calc_fTcoef()
        #self.df = pandas.DataFrame(columns=['Time','V','I','perveance','tauV','tauI','Vreq','Ireq'])
        #df2add = pandas.DataFrame([[self.t,self.V,self.cur,self.perveance_from_cur(self.V,self.cur),self.tauV,self.tauI,0,0]],columns=self.df.columns)
        #self.df = self.df.append(df2add)
        self.history = []
        self.history.append([self.t,self.V,self.cur,self.perveance_from_cur(self.V,self.cur),self.tauV,self.tauI,0,0])
        self.history_columns = ['Time','V','I','perveance','tauV','tauI','Vreq','Ireq']

    def calc_fTcoef(self):
        ac1 = self.fc_coef[0][0]
        ac2 = self.fc_coef[1][0]
        bc1 = self.fc_coef[0][1]
        bc2 = self.fc_coef[1][1]
        cc1 = self.fc_coef[0][2]
        cc2 = self.fc_coef[1][2]

        self.ft_ap = ac1 + ac2 * np.sqrt(2.0) / 2.0 + (1.0 - ac1 - ac2) * np.sqrt(3.0) / 3.0
        self.ft_bp = (bc1 + bc2 * np.sqrt(2.0) / 2.0 - (bc1 + bc2) * np.sqrt(3.0) / 3.0) / 1000
        self.ft_cp = (cc1 + cc2 * np.sqrt(2.0) / 2.0 - (cc1 + cc2) * np.sqrt(3.0) / 3.0) / 1000000
        self.ft_dp = (ac1 + ac2 / 2.0 + 1.0 / 3.0 - ac1 / 3.0 - ac2 / 3.0)
        self.ft_ep = (bc1 + bc2 / 2.0 - bc2 / 3.0 - bc1 / 3.0) / 1000
        self.ft_fp = (cc1 + cc2 / 2.0 - cc2 / 3.0 - cc1 / 3.0) / 1000000

        self.ft_ad = (self.ft_bp * self.ft_dp - self.ft_ep * self.ft_ap)
        self.ft_bd = (
            self.ft_bp * self.ft_ep + 2 * self.ft_cp * self.ft_dp - self.ft_ep * self.ft_bp - 2 * self.ft_ap * self.ft_fp)
        self.ft_cd = (-self.ft_bp * self.ft_fp + self.ft_ep * self.ft_cp)
        self.ft_dd = self.ft_dp * self.ft_dp
        self.ft_ed = 2 * self.ft_dp * self.ft_ep
        self.ft_fd = (2 * self.ft_dp * self.ft_fp + self.ft_ep * self.ft_ep)
        self.ft_gd = 2 * self.ft_ep * self.ft_fp
        self.ft_hd = self.ft_fp * self.ft_fp

    def calc_epsNV(self, V):
        return self.epsN_coeff[0] * V / 1000 + self.epsN_coeff[1]

    def calc_epsT(self, k):
        return (self.epsT_coeff[0] * k * k + self.epsT_coeff[1] * k + self.epsT_coeff[2])

    def cur_from_perveance(self, V, k, micropervs=True):
        scale = 1
        if micropervs:
            scale = 1000000
        return k * V ** (1.5) / scale

    def perveance_from_cur(self, V, cur, micropervs=True):
        scale = 1
        if micropervs:
            scale = 1000000
        return cur * scale / (V ** (1.5))

    def calc_qbeam(self, cur):
        return 0.8435 + 0.0336 * cur

    def calc_qtot(self, cur):
        return self.qNZ + self.qSrc - self.calc_qbeam(cur)

    def calc_epsNq(self, cur):
        return 1.0 - np.exp(-0.1034 * self.calc_qtot(cur))

    def calc_power(self, V, k, duty_cycle=1):
        cur = self.cur_from_perveance(V, k)
        power = duty_cycle * V * cur * self.calc_epsNV(V) * self.calc_epsNq(cur) * self.calc_epsT(
            k) * self.dd * self.RL * self.ol
        return power

    def calc_fT(self, V):
        return (self.ft_ap + self.ft_bp * V + self.ft_cp * V * V) / (self.ft_dp + self.ft_ep * V + self.ft_fp * V * V)

    def calc_torque(self, V, k, duty_cycle=1):
        power = self.calc_power(V, k, duty_cycle)
        if V == 0:
            return 0
        return self.calc_fT(V) * self.etaT * power * V ** -0.5

    def calc_derivatives(self, V, k, duty_cycle=1):
        depsN_dV = self.epsN_coeff[0] / 1000.0
        p = k / 1000000.0
        cur = self.cur_from_perveance(V, k)
        depsT_dp = 2 * self.epsT_coeff[0] * p * 10 ** 12.0 + self.epsT_coeff[1] * 10.0 ** 6.0
        dfT_dV = (self.ft_ad + self.ft_bd * V + self.ft_cd * V * V) / (
            self.ft_dd + self.ft_ed * V + self.ft_fd * V * V + self.ft_gd * V * V * V + self.ft_hd * V * V * V * V)
        detap_dV = -self.RL * self.dd * self.ol * 1.5 * 0.00347424 * p * V ** 0.5 * np.exp(
            -0.1034 * (self.calc_qtot(cur)))
        detap_dp = -self.RL * self.dd * self.ol * 0.00347424 * V ** 1.5 * np.exp(-0.1034 * (self.calc_qtot(cur)))

        dpower_dtheta = self.calc_power(V, k, duty_cycle=1)
        dtorque_dtheta = self.calc_torque(V, k, duty_cycle=1)

        etap = self.RL * self.dd * self.ol * self.calc_epsNq(cur)
        epsN = self.calc_epsNV(V)
        epsT = self.calc_epsT(k)
        dpower_dV = 2.5 * etap * duty_cycle * epsN * epsT * p * V ** 1.5 + depsN_dV * etap * duty_cycle * epsT * p * V ** 2.5 + detap_dV * duty_cycle * epsN * epsT * p * V ** 2.5

        P = self.calc_power(V, k, duty_cycle)
        fT = self.calc_fT(V)
        dtorque_dV = -0.5 * self.etaT * fT * P * V ** (
            -1.5) + self.etaT * fT * V ** -0.5 * dpower_dV + self.etaT * P * V ** -0.5 * dfT_dV

        dpower_dp = etap * duty_cycle * epsN * epsT * V ** (
            2.5) + depsT_dp * etap * duty_cycle * epsN * p * V ** 2.5 + detap_dp * duty_cycle * epsN * epsT * p * V ** (
        2.5)
        dtorque_dp = fT * self.etaT * dpower_dp * V ** (-0.5)

        derivatives = {'dpower_dtheta': dpower_dtheta, 'dpower_dV': dpower_dV, 'dpower_dp': dpower_dp,
                       'dtorque_dtheta': dtorque_dtheta, 'dtorque_dV': dtorque_dV, 'dtorque_dp': dtorque_dp}
        return derivatives

    def check_theta_derivatives(self, V, k):
        thetarange = np.linspace(0.0, 1.0, 40)
        power = [self.calc_power(V, k, theta) for theta in thetarange]
        torque = [self.calc_torque(V, k, theta) for theta in thetarange]
        derivatives = [self.calc_derivatives(V, k, theta) for theta in thetarange]
        dtorque_dtheta = [derivative['dtorque_dtheta'] for derivative in derivatives]
        plt.plot(thetarange, torque)
        plt.plot(thetarange, dtorque_dtheta)
        dpower_dtheta = [derivative['dpower_dtheta'] for derivative in derivatives]
        plt.figure()
        plt.plot(thetarange, power)
        plt.plot(thetarange, dpower_dtheta)

    def check_voltage_derivatives(self, k, theta):
        Vrange = np.linspace(45000, 90000, 300)
        power = [self.calc_power(V, k, theta) for V in Vrange]
        torque = [self.calc_torque(V, k, theta) for V in Vrange]
        derivatives = [self.calc_derivatives(V, k, theta) for V in Vrange]
        dtorque_dV = [derivative['dtorque_dV'] for derivative in derivatives]
        plt.plot(Vrange, np.append(np.diff(torque) / np.diff(Vrange), 0))
        plt.plot(Vrange, dtorque_dV)
        dpower_dV = [derivative['dpower_dV'] for derivative in derivatives]
        plt.figure()
        plt.plot(Vrange, np.append(np.diff(power) / np.diff(Vrange), 0))
        plt.plot(Vrange, dpower_dV)

    def check_p_derivatives(self, V, theta):
        krange = np.linspace(1.8, 3.0, 300)
        prange = krange * 10.0 ** -6.0
        power = [self.calc_power(V, k, theta) for k in krange]
        torque = [self.calc_torque(V, k, theta) for k in krange]
        derivatives = [self.calc_derivatives(V, k, theta) for k in krange]
        dtorque_dp = [derivative['dtorque_dp'] for derivative in derivatives]
        plt.plot(prange, np.append(np.diff(torque) / np.diff(prange), 0))
        plt.plot(prange, dtorque_dp)
        dpower_dp = [derivative['dpower_dp'] for derivative in derivatives]
        plt.figure()
        plt.plot(prange, np.append(np.diff(power) / np.diff(prange), 0))
        plt.plot(prange, dpower_dp)

    def update(self, time_next, input=[0, 0]):
        #input[0] = offset to nominal beam voltage
        #input[1] = offset to nominal Langmuir probe voltage target
        dt = time_next - self.t
        self.t = time_next
        self.V = self.V + dt * (-self.V + input[0] + self.V0) / self.tauV
        self.cur = self.cur + dt * (-self.cur + self.cur0 + self.kprobe*input[1]) / self.tauI
        #df2add = pandas.DataFrame([[self.t, self.V, self.cur, self.perveance_from_cur(self.V, self.cur), self.tauV, self.tauI, input[0], input[1]]], columns=self.df.columns)
        df2add = [self.t, self.V, self.cur, self.perveance_from_cur(self.V, self.cur), self.tauV,
                                    self.tauI, input[0], input[1]]

        #self.df = pandas.concat([self.df,df2add])
        self.history.append(df2add)

    def terminate(self):
        self.df = pandas.DataFrame(self.history,columns=self.history_columns)
        self.history = []
        return 0

    def output(self, noise=False, sigma_V=0.1, sigma_cur=0.1):
        noise_V = 0.0
        noise_cur = 0.0
        if noise:
            noise_V = np.random.normal(0,sigma_V, 1)
            noise_cur = np.random.normal(0, sigma_cur, 1)
        return np.array([self.V+noise_V,self.cur+noise_cur])

    def self_test(self, V, k, duty_cycle, check_dynamics=False,check_derivatives=False):
        print('Name: ' + self.name)
        cur = self.cur_from_perveance(V, k)
        print(cur)
        print('epsT: ' + str(self.calc_epsT(k)))
        print(self.calc_qbeam(cur))
        print(self.calc_qtot(cur))
        print('epsNV: ' + str(self.calc_epsNV(V)))
        print('epsNq: ' + str(self.calc_epsNq(cur)))
        print('power: ' + str(self.calc_power(V, k, duty_cycle)))
        print('etaT: ' + str(self.etaT))
        print('fT: ' + str(self.calc_fT(V)))
        print('torque: ' + str(self.calc_torque(V, k, duty_cycle)))

        if check_dynamics:
            t = np.linspace(0, 1, 500)
            i = interp1d([0, 0.5, 0.75, 1], [45000, 50000, 55000, 55000], kind='zero', fill_value=0)
            [self.update(t1, [i(t1), i(t1)/2000]) for t1 in t]
            plt.plot(self.df['Time'],self.df['V'])
            plt.figure()
            plt.plot(self.df['Time'], self.df['I'])
            plt.figure()
            plt.plot(self.df['Time'], self.df['perveance'])

        if check_derivatives:
            self.check_p_derivatives(V, duty_cycle)
            self.check_voltage_derivatives(k, duty_cycle)
            self.check_theta_derivatives(V, k)


def prepare_d3d_beams(Rtans=None):
    if Rtans is None:
        Rtans = {
            '30L': 1.149,
            '30R': 0.749,
            '150L': 1.149,
            '150R': 0.749,
            '210L': -0.749,
            '210R': -1.149,
            '330L': 1.149,
            '330R': 0.749,
        }
    beams = collections.OrderedDict()
    beams['30L'] = Beam(
            name='30L',
            epsT_coeff=[-17, 115, -107],
            qNZ=13,
            qSrc=11,
            Rtan=Rtans['30L'],
            ol=1,
            kprobe=1.85,
            optimal_perveances=np.array([2.9,2.9,2.9]))
    beams['30R'] = Beam(
            name='30R', epsT_coeff=[-89.85, 463.090, -510.540],
            qNZ=13,
            qSrc=11,
            Rtan=Rtans['30R'],
            ol=1,
            kprobe=2.0,
            optimal_perveances=np.array([2.45,2.45,2.45]))
    beams['150L'] = Beam(
            name='150L', epsT_coeff=[-44.42, 227.21, -212.640],
            qNZ=13,
            qSrc=11,
            Rtan=Rtans['150L'],
            ol=1,
            kprobe=1.82,
            optimal_perveances=np.array([2.19,2.245,2.29]))
    beams['150R'] = Beam(
            name='150R',
            epsT_coeff=[-33.59, 187.27, -180.610],
            qNZ=13,
            qSrc=11,
            Rtan=Rtans['150R'],
            ol=1,
            kprobe=1.81,
            optimal_perveances=np.array([2.25,2.305,2.35]))
    beams['210L'] = Beam(
            name='210L',
            epsT_coeff=[-27.51, 154.36, -123.15],
            qNZ=13,
            qSrc=11,
            Rtan=Rtans['210L'],
            ol=1,
            kprobe=2.01,
            optimal_perveances=np.array([2.32,2.385,2.42]))
    beams['210R'] = Beam(
            name='210R',
            epsT_coeff=[-16.73, 104.88, -78.16],
            qNZ=13,
            qSrc=11,
            Rtan=Rtans['210R'],
            ol=1,
            kprobe=2.01,
            optimal_perveances=np.array([2.25,2.29,2.35]))
    beams['330L'] = Beam(
            name='330L',
            epsT_coeff=[-33.91, 212.93, -237.99],
            qNZ=13,
            qSrc=11,
            Rtan=Rtans['330L'],
            ol=1,
            kprobe=2.07,
            optimal_perveances=np.array([2.79,2.84,2.88]))
    beams['330R'] = Beam(
            name='330R',
            epsT_coeff=[-24.09, 143.090, -126.57],
            qNZ=13,
            qSrc=11,
            Rtan=Rtans['330R'],
            ol=1,
            kprobe=1.83,
            optimal_perveances=np.array([2.67,2.75,2.75]))
    return beams

def update_beams(beams,t1,delta_Vs,delta_Vprobes):
    [beams[beam].update(t1, input=[delta_Vs[beam], delta_Vprobes[beam]]) for beam
     in beams.keys()]

def terminate_beams(beams):
    [beams[beam].terminate() for beam in beams.keys()]

def output_beams(beams,noise=True,sigma_V=10):
    pinj = 0
    tinj = 0
    beam_voltage_meas = np.zeros((8))
    beam_current_meas = np.zeros((8))
    for j, beam in enumerate(beams.keys()):
        output = beams[beam].output(noise=noise, sigma_V=sigma_V)
        k = beams[beam].perveance_from_cur(beams[beam].V, beams[beam].cur)
        pinj = pinj + beams[beam].calc_power(beams[beam].V, k, duty_cycle=1)
        tbeam = beams[beam].calc_torque(beams[beam].V, k, duty_cycle=1)
        tinj = tinj + tbeam
        beam_voltage_meas[j] = output[0]
        beam_current_meas[j] = output[1]
    return pinj,tinj,beam_voltage_meas,beam_current_meas

def plot_ranges():
    beams = prepare_d3d_beams()
    # todo: compare ranges for two beams, show there is some nonlinearity and differences between beams
    # todo: plot v150R vs v150L contours of power and torque
    beam = beams['150R']
    Vrange = np.arange(45000, 90000, 40)
    plt.figure()
    for k in [1.8, 2.3, 2.8]:
        power = [beam.calc_power(V, k) for V in Vrange]
        plt.plot(Vrange, power)

    plt.figure()
    for k in [1.8, 2.3, 2.8]:
        torque = [beam.calc_torque(V, k) for V in Vrange]
        plt.plot(Vrange, torque)

        # todo: get voltages and currents from shots and calculate powers, torques
        # todo: add dynamic model for response to voltage and current requests
