from scipy.optimize import fsolve
from math import cos, pi
import yaml


class Property:
    def __init__(self, yml_path):
        self.params = yaml.safe_load(open(yml_path, encoding="utf-8").read())

    def __getattr__(self, item):
        return self.params.get(item, None)


class SMA(object):
    def __init__(self, property_path, eps_0, sigma_0, T0, ksi_s0, ksi_T0):
        sma_property = Property(property_path)
        self.load = True
        # 材料常数
        self.Da = sma_property.Da
        self.Dm = sma_property.Dm
        self.Mf = sma_property.Mf
        self.Ms = sma_property.Ms
        self.As = sma_property.As
        self.Af = sma_property.Af
        self.eps_l = sma_property.eps_l
        self.Cm = sma_property.Cm
        self.Ca = sma_property.Ca
        self.sigma_s = sma_property.sigma_s
        self.sigma_f = sma_property.sigma_f
        self.theta = sma_property.theta
        # 计算得到的材料常数
        self.aM = pi / (self.Ms - self.Mf)
        self.aA = pi / (self.Af - self.As)
        # 初始条件
        self.eps_0 = eps_0
        self.T0 = T0
        self.sigma_0 = sigma_0
        self.ksi_s0 = ksi_s0
        self.ksi_T0 = ksi_T0
        # 计算得到的初始条件
        self.ksi_0 = self.ksi_s0 + self.ksi_T0
        self.D0 = self.ksi_0 * self.Dm + (1 - self.ksi_0) * self.Da
        self.omega_0 = -self.eps_l * self.D0

    def calc_by_eps(self, eps):
        func = lambda x: self.calc_by_sigma(x)[0] - eps
        sigma = fsolve(func, self.sigma_0).item()
        eps, sigma, T, ksi_s, ksi_T = self.calc_by_sigma(sigma)
        return eps, sigma, T, ksi_s, ksi_T
    
    def calc_by_T(self, T):
        func = lambda x: self.calc_by_sigma(x, T)[0] - self.eps_0
        sigma = fsolve(func, self.sigma_0).item()
        eps, sigma, T, ksi_s, ksi_T = self.calc_by_sigma(sigma, T)
        return eps, sigma, T, ksi_s, ksi_T

    def calc_by_sigma(self, sigma, T=None):
        if T is None:
            T = self.T0
        ksi_s, ksi_T = self._get_ksi(sigma, T)
        ksi = ksi_s + ksi_T
        D = ksi * self.Dm + (1 - ksi) * self.Da
        omega = -self.eps_l * D
        eps = (sigma - self.sigma_0 - (
                omega * ksi_s - self.omega_0 * self.ksi_s0) + self.D0 * self.eps_0 - self.theta * (T - self.T0)) / D
        return eps, sigma, T, ksi_s, ksi_T

    def _get_th(self, T):
        sigma_bias = self.Cm * (T - self.Ms)
        sigma_inf_A = max(self.Ca * (T - self.Af), 0)
        sigma_sup_A = max(self.Ca * (T - self.As), 0)
        sigma_inf_M = max(self.sigma_s + sigma_bias, self.sigma_s)
        sigma_sup_M = max(self.sigma_f + sigma_bias, self.sigma_f)
        return sigma_inf_A, sigma_sup_A, sigma_inf_M, sigma_sup_M

    def _get_ksi(self, sigma, T):
        sigma_inf_A, sigma_sup_A, sigma_inf_M, sigma_sup_M = self._get_th(T)
        if sigma < sigma_sup_A:
            if self.ksi_0 == 0:
                ksi_s, ksi_T = 0, 0
            else:
                if sigma < sigma_inf_A:
                    phi = self.aA * (self.Af - self.As)
                else:
                    phi = self.aA * (T - self.As - sigma / self.Ca)
                ksi = self.ksi_0 / 2 * (cos(phi) + 1)
                ksi_s = self.ksi_s0 * ksi / self.ksi_0
                ksi_T = self.ksi_T0 * ksi / self.ksi_0
        else:
            delta_T_ksi = 0
            if sigma < sigma_inf_M:
                ksi_s = self.ksi_s0
            elif sigma_inf_M <= sigma < sigma_sup_M:
                phi = pi / (sigma_sup_M - sigma_inf_M) * (sigma - sigma_sup_M)
                ksi_s = (1 - self.ksi_s0) / 2 * cos(phi) + (1 + self.ksi_s0) / 2
            else:
                ksi_s = 1
                if self.Mf < T < self.Ms and T < self.T0:
                    delta_T_ksi = (1 - self.ksi_T0) / 2 * (cos(self.aM * (T - self.Mf)) + 1)
            ksi_T = self.ksi_T0 * (1 - ksi_s) / (1 - self.ksi_s0) + delta_T_ksi if self.ksi_T0 != 0 else 0
        
        if not self.load:
            ksi_s = min(ksi_s, self.ksi_s0)

        return ksi_s, ksi_T

    def _update_initial(self):
        self.ksi_0 = self.ksi_s0 + self.ksi_T0
        self.D0 = self.ksi_0 * self.Dm + (1 - self.ksi_0) * self.Da
        self.omega_0 = -self.eps_l * self.D0

    def update(self, eps_0, sigma_0, T0, ksi_s0, ksi_T0):
        self.eps_0, self.sigma_0, self.T0, self.ksi_s0, self.ksi_T0 = eps_0, sigma_0, T0, ksi_s0, ksi_T0
        self._update_initial()
