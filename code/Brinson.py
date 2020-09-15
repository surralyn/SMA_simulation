from math import cos, sin, pi
import numpy as np
from scipy.optimize import fsolve
import yaml


class Property:
    def __init__(self, yml_path):
        self.params = yaml.safe_load(open(yml_path, encoding="utf-8").read())

    def __getattr__(self, item):
        return self.params.get(item, None)


class SMA(object):
    def __init__(self, property_path=None):
        if property_path:
            sma_property = Property(property_path)
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
            self.delta_eps_j = sma_property.delta_eps_j
            self.eps_0 = sma_property.eps_0
        else:
            self.Da = 67000
            self.Dm = 26300
            self.Mf = 9
            self.Ms = 18.4
            self.As = 34.5
            self.Af = 49
            self.eps_l = 0.067
            self.Cm = 8
            self.Ca = 13.8
            self.sigma_s = 100
            self.sigma_f = 170
            self.delta_eps_j = 0.00001
            self.eps_0 = 0

    def curve(self, T, eps_max):
        Da = self.Da
        Dm = self.Dm
        Mf = self.Mf
        Ms = self.Ms
        As = self.As
        Af = self.Af
        eps_l = self.eps_l
        Cm = self.Cm
        Ca = self.Ca
        sigma_s = self.sigma_s
        sigma_f = self.sigma_f
        delta_eps_j = self.delta_eps_j
        eps_0 = self.eps_0

        # 给定加载段控制点值
        sigma_ms = sigma_s + Cm * (T - Ms)  # 根据 Brinson 本构模型得到
        sigma_mf = sigma_f + Cm * (T - Ms)  # 根据 Brinson 本构模型得到的应力区间上限值
        eps_ms = sigma_ms / Da  # 根据 Brinson 本构模型得到

        a1, a2, b1, b2, c1, c2 = [], [], [], [], [], []
        if eps_ms < eps_max:
            # 定义应力应变区间
            # 通过加载段 brinson 模型方程组（初始马氏体含量为 0）求解对应输入值 eps_max的应力幅值
            # 加载段应力应变关系模拟
            step = abs(int((eps_max - eps_0) // delta_eps_j))
            for eps_i in np.linspace(eps_0, eps_max, step):  # 进入加载段循环
                if eps_i <= eps_ms:  # 判断第 i 点应变是否属于加载的第一阶段（奥氏体相下的弹性加载段）
                    sigma_i = eps_i * Da  # 加载第一阶段的应力应变关系
                    xi_i = 0  # 加载第一阶段为发生相变，故马氏体含量为零
                    a1.append(eps_i)  # 提取此段的所有应变点值
                    b1.append(sigma_i)  # 提取此段的所有应力点值
                    c1.append(xi_i)  # 提取此段的所有马氏体含量值
                elif eps_ms < eps_i <= eps_max:  # 判断第 i 点应变是否属于加载的第二阶段（马氏体正相变阶段）
                    f1 = F1(Da, Dm, Cm, T, Ms, sigma_s, sigma_f, eps_i, eps_l)
                    # 满足 Brinson 本构模型的正相变阶段应力应变关系
                    point = np.array((sigma_ms + sigma_mf) / 2)
                    sigma_i = fsolve(f1, point).item()  # 解超越方程得相应的各应力点值
                    xi_i = 0.5 * cos((pi / (sigma_s - sigma_f)) * (
                            sigma_i - sigma_f - Cm * (T - Ms))) + 0.5  # 对应第 i 点的马氏体含量
                    a2.append(eps_i)  # 提取此段的所有应变点值
                    b2.append(sigma_i)  # 提取此段的所有应力点值
                    c2.append(xi_i)  # 提取此段的所有马氏体含量值
                else:
                    raise BaseException('加载阶段判断出界')

            p_j = a1 + a2  # 提取整个加载段的所有应变点值
            t_j = b1 + b2  # 提取整个加载段的所有应力点值
            q_j = c1 + c2  # 提取整个加载段的所有马氏体含量值

        elif 0 <= eps_max <= eps_ms:
            step = abs(int((eps_max - eps_0) // delta_eps_j))
            for eps_i in np.linspace(eps_0, eps_max, step):  # 判断第 i 点应变是否属于卸载第三阶段（母相奥氏体态下的弹性恢复阶段）
                sigma_i = Da * eps_i  # 弹性恢复下的应力应变关系
                xi_i = 0
                a1.append(eps_i)  # 提取此段的所有应变点值
                b1.append(sigma_i)  # 提取此段的所有应力点值
                c1.append(xi_i)  # 提取此段的所有马氏体含量值
            p_j = a1  # 提取整个加载段的所有应变点值
            t_j = b1  # 提取整个加载段的所有应力点值
            q_j = c1  # 提取整个加载段的所有马氏体含量值
        else:
            raise BaseException('不属于加载一二阶段和卸载第三阶段')

        # 卸载阶段
        # 卸载段初值
        sigma_max = max(t_j)  # 提取加载结束点的应力值
        xi_max = max(q_j)  # 提取加载结束点的马氏体含量值
        D_max = (Da + xi_max * (Dm - Da))  # 对应加载结束点马氏体含量的弹模

        # 给定卸载段步长、步数
        delta_eps_x = 0.00001  # 指定步长

        a5, b5, c5 = [], [], []
        if eps_ms < eps_max:
            f3 = F3(Af, As, Ca, Da, Dm, T, eps_l, xi_max)
            sigma_i3 = fsolve(f3, np.array(75.0)).item()
            sigma_i4 = fsolve(f3, np.array(250.0)).item()  #
            sigma_i5 = fsolve(f3, np.array(450.0)).item()  #
            sigma_i1 = (sigma_i3 + sigma_i4) / 2  # 逆相变结束点控制应力值
            eps_i1 = sigma_i1 / (
                    Da + 0.5 * xi_max * (cos(pi * (Ca * T - Ca * As - sigma_i1) / (Ca * Af - Ca * As)) + 1) * (
                    Dm - Da)) + eps_l * 0.5 * xi_max * (
                             cos(pi * (Ca * T - Ca * As - sigma_i1) / (Ca * Af - Ca * As)) + 1)
            # 对应逆相变结束点的应变值
            sigma_i2 = (sigma_i4 + sigma_i5) / 2  # 逆相变开始点控制应力值
            eps_i2 = sigma_i2 / (
                    Da + 0.5 * xi_max * (cos(pi * (Ca * T - Ca * As - sigma_i2) / (Ca * Af - Ca * As)) + 1) * (
                    Dm - Da)) + eps_l * 0.5 * xi_max * (
                             cos(pi * (Ca * T - Ca * As - sigma_i2) / (Ca * Af - Ca * As)) + 1)
            # 对应逆相变开始点的应变值
            # 卸载循环
            a3, b3, c3, a4, b4, c4 = [], [], [], [], [], []
            step = abs(int((eps_max - eps_i1) // delta_eps_x))
            for eps_i in np.linspace(eps_i1, eps_max, step):  # 进入卸载段循环
                if eps_i2 < eps_i <= eps_max:  # 判断第 i 点应变是否属于卸载第一阶段（混合相下的弹性恢复段）
                    sigma_i = D_max * (eps_i - eps_max) + sigma_max  # 弹性恢复下的应力应变关系
                    xi_i = xi_max  # 由于未进入马氏体逆相变阶段，故马氏体含量保持不变
                    a3.append(eps_i)  # 提取此段的所有应变点值
                    b3.append(sigma_i)  # 提取此段的所有应力点值
                    c3.append(xi_i)  # 提取此段的所有马氏体含量值
                elif eps_i1 <= eps_i <= eps_i2:  # 判断第 i 点应变是否属于卸载第二阶段（马氏体逆相变恢复阶段）
                    f2 = F2(As, Af, Ca, Da, Dm, T, eps_i, eps_l, xi_max)
                    # 满足 Brinson 本构模型的逆相变阶段应力应变关系
                    point = np.array((sigma_i1 + sigma_i2) / 2)
                    sigma_i = fsolve(f2, point).item()  # 解 超越方程得相应的各应力点值
                    xi_i = 0.5 * xi_max * (cos(
                        pi * (Ca * T - Ca * As - sigma_i) / (Ca * Af - Ca * As)) + 1)  # 对应第 i 点的卸载段马氏体含量
                    a4.append(eps_i)  # 提取此段的所有应变点值
                    b4.append(sigma_i)  # 提取此段的所有应力点值
                    c4.append(xi_i)  # 提取此段的所有马氏体含量值
                else:
                    raise BaseException('卸载循环判断出界')

            eps_00 = min(a4)
            sigma_00 = min(b4)
            xi_00 = min(c4)
            D_00 = (Da + xi_00 * (Dm - Da))
            eps_01 = (0 - sigma_00) / D_00 + eps_00  # sigma_i=0 时对应的 eps_i 是多少

            step = abs(int((eps_i1 - eps_01) // delta_eps_x))
            for eps_i in np.linspace(eps_01, eps_i1, step):  # 判断第 i 点应变是否属于卸载第三阶段（母相奥氏体态下的弹性恢复阶段）
                sigma_i = D_00 * (eps_i - eps_00) + sigma_00  # 弹性恢复下的应力应变关系
                xi_i = xi_00  #
                a5.append(eps_i)  # 提取此段的所有应变点值
                b5.append(sigma_i)  # 提取此段的所有应力点值
                c5.append(xi_i)  # 提取此段的所有马氏体含量值

            p_x = a3[::-1] + a4[::-1] + a5[::-1]
            t_x = b3[::-1] + b4[::-1] + b5[::-1]
            q_x = c3[::-1] + c4[::-1] + c5[::-1]

        elif 0 <= eps_max <= eps_ms:
            step = abs(int(eps_max // delta_eps_x))
            for eps_i in np.linspace(0, eps_max, step):  # 判断第 i 点应变是否属于卸载第三阶段（母相奥氏体态下的弹性恢复阶段）
                sigma_i = Da * eps_i  # 弹性恢复下的应力应变关系
                xi_i = 0  #
                a5.append(eps_i)  # 提取此段的所有应变点值
                b5.append(sigma_i)  # 提取此段的所有应力点值
                c5.append(xi_i)  # 提取此段的所有马氏体含量值

            p_x = a5[::-1]  # 按卸载顺序提取整个卸载段的所有应变点值
            t_x = b5[::-1]  # 按卸载顺序提取整个卸载段的所有应力点值
            q_x = c5[::-1]  # 按卸载顺序提取整个卸载段的所有马氏体含量值
        else:
            raise BaseException('不属于卸载一二阶段和卸载第三阶段')

        p = np.array(p_j + p_x)  # 提取整个加卸载循环的所有应变点值
        t = np.array(t_j + t_x)  # 提取整个加卸载循环的所有应力点值
        q = np.array(q_j + q_x)  # 提取整个加卸载循环的所有马氏体含量值

        mask = t >= 0
        p, t, q = p[mask], t[mask], q[mask]

        m = {
            'strain': p,
            'stress': t,
            'content': q,
        }

        return m


class F1(object):
    def __init__(self, Da, Dm, Cm, T, Ms, sigma_s, sigma_f, eps_i, eps_l):
        self.Da = Da
        self.Dm = Dm
        self.Cm = Cm
        self.T = T
        self.Ms = Ms
        self.sigma_s = sigma_s
        self.sigma_f = sigma_f
        self.eps_i = eps_i
        self.eps_l = eps_l

    def __call__(self, sigma_i):
        Da = self.Da
        Dm = self.Dm
        Cm = self.Cm
        T = self.T
        Ms = self.Ms
        sigma_s = self.sigma_s
        sigma_f = self.sigma_f
        eps_i = self.eps_i
        eps_l = self.eps_l
        p1 = 0.5 * cos((pi / (sigma_s - sigma_f)) * (sigma_i - sigma_f - Cm * (T - Ms))) + 0.5
        r = sigma_i - (Da + p1 * (Dm - Da)) * (eps_i - eps_l * p1)
        return r


class F2(object):
    def __init__(self, As, Af, Ca, Da, Dm, T, eps_i, eps_l, xi_max):
        self.As = As
        self.Af = Af
        self.Ca = Ca
        self.Da = Da
        self.Dm = Dm
        self.T = T
        self.eps_i = eps_i
        self.eps_l = eps_l
        self.xi_max = xi_max

    def __call__(self, sigma_i):
        As = self.As
        Af = self.Af
        Ca = self.Ca
        Da = self.Da
        Dm = self.Dm
        T = self.T
        eps_i = self.eps_i
        eps_l = self.eps_l
        xi_max = self.xi_max
        p1 = cos(pi * (Ca * T - Ca * As - sigma_i) / (Ca * Af - Ca * As)) + 1
        r = sigma_i - (Da + 0.5 * xi_max * p1 * (Dm - Da)) * (eps_i - eps_l * 0.5 * xi_max * p1)
        return r


class F3(object):
    def __init__(self, Af, As, Ca, Da, Dm, T, eps_l, xi_max):
        self.Af = Af
        self.As = As
        self.Ca = Ca
        self.Da = Da
        self.Dm = Dm
        self.T = T
        self.eps_l = eps_l
        self.xi_max = xi_max

    def __call__(self, sigma_i):
        Af = self.Af
        As = self.As
        Ca = self.Ca
        Da = self.Da
        Dm = self.Dm
        T = self.T
        eps_l = self.eps_l
        xi_max = self.xi_max
        p0 = Ca * Af - Ca * As
        p1 = pi * (Ca * T - Ca * As - sigma_i) / p0
        p2 = -0.5 * xi_max * (pi / p0) ** 2 * cos(p1)
        p3 = Da + 0.5 * xi_max * (cos(p1) + 1)
        p4 = (Dm - Da) * 0.5 * xi_max * pi / p0 * sin(p1)
        p5 = p3 * (Dm - Da)
        r = (-2 * p4 - sigma_i * (Dm - Da) * p2) / p5 ** 2 + 2 * sigma_i * p4 ** 2 / p5 ** 3 + eps_l * p2
        return r
