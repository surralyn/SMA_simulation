import math
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
            self.D = sma_property.D
            self.Mf = sma_property.Mf
            self.Ms = sma_property.Ms
            self.As = sma_property.As
            self.Af = sma_property.Af
            self.eps_l = sma_property.eps_l
            self.Cm = sma_property.Cm
            self.Ca = sma_property.Ca
            self.delta_eps_j = sma_property.delta_eps_j
            self.eps_0 = sma_property.eps_0
            self.delta_eps_x = sma_property.delta_eps_x
        else:
            self.D = 46650
            self.Mf = 9
            self.Ms = 18.4
            self.As = 34.5
            self.Af = 49
            self.eps_l = 0.067
            self.Cm = 10.3
            self.Ca = 10.3
            self.delta_eps_j = 0.00005
            self.eps_0 = 0
            self.delta_eps_x = 0.00005

    def curve(self, T, eps_max):
        D = self.D
        Mf = self.Mf
        Ms = self.Ms
        As = self.As
        Af = self.Af
        eps_l = self.eps_l
        Cm = self.Cm
        Ca = self.Ca
        Aa = math.log(0.01) / (As - Af)
        Am = math.log(0.01) / (Ms - Mf)
        Ba = Aa / Ca
        Bm = Am / Cm

        # 给定加载段控制点值
        sigma_ms = Cm * (T - Ms)  # 根据 tanaka 本构模型得到
        eps_ms = sigma_ms / D  # 根据 tanaka 本构模型得到
        # xi_ms = 0  # 假定材料初始状态为零应力下的完全奥氏体态

        # 给定加载段应变幅值、步长、步数
        delta_eps_j = self.delta_eps_j  # 指定加载段步长
        # steps = int(eps_max // delta_eps_j)  # 加载段步数 n

        # 给定加载的初始值
        eps_0 = self.eps_0  # 假定材料初始状态下的应力应变为零
        # sigma_0 = 0

        # 定义应力应变区间
        sigma_mf = math.log(0.01) / Bm + Cm * (T - Ms)  # 根据 tanaka 本构模型得到的应力区间上限值
        # f0 = lambda sigma_i, eps_i: sigma_i - D * (eps_i - eps_l * (1 - math.exp(Am * (Ms - T) + Bm * sigma_i)))
        # point = np.array((sigma_ms + sigma_mf) / 2)
        # sigma_max = fsolve(lambda sigma_i: f0(sigma_i, eps_max), point)
        # 通过加载段 tanaka 模型方程组（初始马氏体含量为 0）求解对应输入值 eps_max 的应力幅值
        # eps_i=[0,eps_max]
        # sigma_i=[0,sigma_max]  #本程序的应力应变变化区间

        # 加载段应力应变关系模拟
        i = 1
        a1, a2, b1, b2, c1, c2 = [], [], [], [], [], []
        for eps_i in np.linspace(eps_0, eps_max, int((eps_max - eps_0) // delta_eps_j)):  # 进入加载段循环
            if eps_i <= eps_ms:  # 判断第 i 点应变是否属于加载的第一阶段（奥氏体相下的弹性加载段）
                sigma_i = eps_i * D  # 加载第一阶段的应力应变关系
                xi_i = 0  # 加载第一阶段为发生相变，故马氏体含量为零
                a1.append(eps_i)  # 提取此段的所有应变点值
                b1.append(sigma_i)  # 提取此段的所有应力点值
                c1.append(xi_i)  # 提取此段的所有马氏体含量值
            elif eps_ms < eps_i <= eps_max:  # 判断第 i 点应变是否属于加载的第二阶段（马氏体正相变阶段）
                D_i = D  # 对应第 i 点的混合弹模
                f1 = lambda sigma_i, eps_i: sigma_i - D * (eps_i - eps_l * (1 - math.exp(Am * (Ms - T) + Bm * sigma_i)))
                # 满足 tanaka 本构模型的正相变阶段应力应变关系
                t = eps_i  # 解 f1 方程的输入参数（对应此阶段的各已知应变点值）
                point = np.array((sigma_ms + sigma_mf) / 2)
                sigma_i = fsolve(lambda sigma_i: f1(sigma_i, t), point).item()  # 解 超 越方程得相应的各应力点值
                xi_i = (1 - math.exp(Am * (Ms - T) + Bm * sigma_i))  # 对应第 i 点的马氏体含量
                a2.append(eps_i)  # 提取此段的所有应变点值
                b2.append(sigma_i)  # 提取此段的所有应力点值
                c2.append(xi_i)  # 提取此段的所有马氏体含量值
            i = i + 1

        p_j = a1 + a2  # 提取整个加载段的所有应变点值
        t_j = b1 + b2  # 提取整个加载段的所有应力点值
        q_j = c1 + c2  # 提取整个加载段的所有马氏体含量值

        # 卸载段初值
        sigma_max = max(t_j)  # 提取加载结束点的应力值
        xi_max = max(q_j)  # 提取加载结束点的马氏体含量值

        # 给定卸载段步长、步数
        delta_eps_x = self.delta_eps_x  # 指定步长
        steps = int(eps_max // delta_eps_x)  # 卸载步数 n

        sigma_i3 = Ca * (T - As) + math.log(xi_max) / Ba
        eps_i3 = sigma_i3 / D + eps_l * xi_max
        sigma_i4 = math.log(0.01) / Ba + Ca * (T - As)
        eps_i4 = sigma_i4 / D

        # 卸载循环
        i = 1
        a3, a4, b3, b4, c3, c4 = [], [], [], [], [], []
        for eps_i in np.linspace(eps_i4, eps_max, int((eps_max - eps_i4) // delta_eps_x)):  # 进入卸载段循环
            if eps_i3 < eps_i <= eps_max:  # 判断第 i 点应变是否属于卸载第一阶段（混合相下的弹性恢复段）
                sigma_i = D * (eps_i - eps_max) + sigma_max  # 弹性恢复下的应力应变关系
                xi_i = xi_max  # 由于未进入马氏体逆相变阶段，故马氏体含量保持不变
                a3.append(eps_i)  # 提取此段的所有应变点值
                b3.append(sigma_i.item())  # 提取此段的所有应力点值
                c3.append(xi_i)  # 提取此段的所有马氏体含量值
            elif eps_i4 <= eps_i <= eps_i3:  # 判断第 i 点应变是否属于卸载第二阶段（马氏体逆相变恢复阶段）
                f2 = lambda sigma_i, eps_i: sigma_i - D * (eps_i - eps_l * math.exp(Aa * (As - T) + Ba * sigma_i))
                # 满足 tanaka 本构模型的逆相变阶段应力应变关系
                p = eps_i  # 解 f2 方程的输入参数（对应此阶段的各已知应变点值）
                sigma_i = fsolve(lambda sigma_i: f2(sigma_i, p), np.array(sigma_i3)).item()  # 解超越方程得相应的各应力点值
                xi_i = math.exp(Aa * (As - T) + Ba * sigma_i)  # 对应第 i 点的卸载段马氏体含量
                a4.append(eps_i)  # 提取此段的所有应变点值
                b4.append(sigma_i)  # 提取此段的所有应力点值
                c4.append(xi_i)  # 提取此段的所有马氏体含量值
            i = i + 1

        p4 = a4
        t4 = b4
        q4 = c4
        eps_00 = min(p4)
        sigma_00 = min(t4)
        xi_00 = min(q4)
        eps_01 = (0 - sigma_00) / D + eps_00  # sigma_i=0 时对应的 eps_i 是多少
        # eps_01<=eps_i&eps_i<eps_i4

        i = 1
        a5, b5, c5 = [], [], []
        steps = abs(int((eps_i4 - eps_01) // delta_eps_x))
        e1 = min(eps_i4, eps_01)
        e2 = max(eps_i4, eps_01)
        for eps_i in np.linspace(e1, e2, steps):
            sigma_i = D * (eps_i - eps_00) + sigma_00  # 弹性恢复下的应力应变关系
            xi_i = xi_00
            i = i + 1
            a5.append(eps_i.item())  # 提取此段的所有应变点值
            b5.append(sigma_i.item())  # 提取此段的所有应力点值
            c5.append(xi_i)  # 提取此段的所有马氏体含量值

        a_x = a3[::-1] + a4[::-1] + a5[::-1]  # 按卸载顺序提取整个卸载段的所有应变点值
        b_x = b3[::-1] + b4[::-1] + b5[::-1]  # 按卸载顺序提取整个卸载段的所有应力点值
        c_x = c3[::-1] + c4[::-1] + c5[::-1]  # 按卸载顺序提取整个卸载段的所有马氏体含量值
        p_x = a_x[::1]
        t_x = b_x[::1]
        q_x = c_x[::1]
        # 解决 b3 的最小值比 b4 的最大值要小的问题
        t_x.sort(reverse=True)  # 得到整个 y_x 向量内数组的降序排列

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
