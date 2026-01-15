
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import math
import csv
import datetime
import os
from scipy.spatial import KDTree
import matplotlib.animation as animation

import matplotlib as mpl
from scipy.interpolate import CubicSpline
from scipy import signal
# 设置Matplotlib使用TkAgg后端
mpl.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径类型选择变量
PATH_TYPE = 0  # 0: 复杂连续变化曲率路径, 1: 直线+圆弧组合路径

SHOW_ANIMATION = True  # 控制是否显示动画
SHOW_REALTIME_PLOTS = True  # 控制是否显示实时图表
SAVE_RESULT_FILES = True  # 控制是否保存结果文件到result文件夹

SPEED = 0.8  # 车速
BETA = 0.3  # 预瞄距离中的曲率系数
DT = 0.05  # 步长
WINDOW = 10  # 搜索窗口大小
MAX_STEER = 0.6  # 最大转向角
WHEELBASE = 0.144  # 轴距

class AckermannVehicle:
    """阿克曼小车模型，包含完整PID控制"""

    def __init__(self, wheelbase=WHEELBASE,
                 kp_vel=1.0, ki_vel=0.1, kd_vel=0.05,
                 kp_steer=1.0, ki_steer=0.05, kd_steer=0.02):
        # 状态变量 [x, y, yaw, velocity, steering_angle]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.wheelbase = wheelbase  # 轴距
        self.trajectory = []  # 存储轨迹点

        # 车辆尺寸 (米)
        self.length = 0.21
        self.width = 0.18

        # PID参数
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel
        self.kp_steer = kp_steer
        self.ki_steer = ki_steer
        self.kd_steer = kd_steer

        # 用于积分项的误差累积
        self.integral_vel = 0.0
        self.integral_steer = 0.0
        self.prev_error_vel = 0.0
        self.prev_error_steer = 0.0

    def update(self, target_velocity, target_steering, dt=DT):
        """更新车辆状态，使用完整PID控制"""
        x, y, yaw, velocity, steering = self.state

        # 速度控制 - PID
        error_vel = target_velocity - velocity
        self.integral_vel += error_vel * dt
        derivative_vel = (error_vel - self.prev_error_vel) / dt
        control_vel = (self.kp_vel * error_vel +
                       self.ki_vel * self.integral_vel +
                       self.kd_vel * derivative_vel)

        # 限制控制量
        control_vel = np.clip(control_vel, -2.0, 2.0)
        velocity += control_vel * dt
        velocity = np.clip(velocity, -1.0, 1.0)  # 速度限制在1m/s

        # 转向角控制 - PID
        error_steer = target_steering - steering
        self.integral_steer += error_steer * dt
        derivative_steer = (error_steer - self.prev_error_steer) / dt
        control_steer = (self.kp_steer * error_steer +
                         self.ki_steer * self.integral_steer +
                         self.kd_steer * derivative_steer)

        # 限制控制量
        control_steer = np.clip(control_steer, -3.0, 3.0)
        steering += control_steer * dt
        # 限制转向角范围
        max_steer = 0.7  # 约40度
        steering = np.clip(steering, -max_steer, max_steer)

        # 更新位置和朝向
        beta = math.atan(0.5 * math.tan(steering))  # 阿克曼转向近似
        yaw += velocity * math.tan(steering) / self.wheelbase * dt
        x += velocity * math.cos(yaw + beta) * dt
        y += velocity * math.sin(yaw + beta) * dt

        # 归一化角度到[-π, π]
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        # 更新状态
        self.state = np.array([x, y, yaw, velocity, steering])

        # 保存误差用于下一次计算
        self.prev_error_vel = error_vel
        self.prev_error_steer = error_steer

        # 记录轨迹
        self.trajectory.append((x, y, yaw))

class Simulation:

    def __init__(self):
        pass

    def main(self):
        car = AckermannVehicle()
        

    def run(self):
        pass