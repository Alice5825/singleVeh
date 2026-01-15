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
PATH_TYPE = 1  # 0: 复杂连续变化曲率路径, 1: 直线+圆弧组合路径

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
    """阿克曼小车模型"""

    def __init__(self, wheelbase=WHEELBASE):
        # 状态变量 [x, y, yaw, velocity, steering_angle]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.wheelbase = wheelbase  # 轴距
        self.trajectory = []  # 存储轨迹点

        # 车辆尺寸 (米)
        self.length = 0.21
        self.width = 0.18

    def update(self, target_velocity, target_steering, dt=DT):
        """更新车辆状态"""
        x, y, yaw, velocity, steering = self.state

        # 简单的一阶动力学模型
        # 速度控制
        acceleration = 2 * (target_velocity - velocity)  # 简化的PID控制
        velocity += acceleration * dt
        # 限制速度范围
        velocity = np.clip(velocity, -1.0, 1.0)  # 速度限制在1m/s

        # 转向角控制
        steering_rate = np.clip(5 * (target_steering - steering),-2,2)  # 简化的PID控制
        steering += steering_rate * dt
        # 限制转向角范围
        max_steer = 0.7  # 约40度，实际是40度到45度之间
        steering = np.clip(steering, -max_steer, max_steer)

        # 更新位置和朝向
        beta = math.atan(0.5 * math.tan(steering))  # 对于阿克曼转向，近似β=l_r/(l_f+l_r)*tan(δ) 0.0068 / 0.144
        yaw += velocity * math.tan(steering) / self.wheelbase * dt  # 注意（math.tan(steering) / self.wheelbase）是半径R
        x += velocity * math.cos(yaw + beta) * dt
        y += velocity * math.sin(yaw + beta) * dt

        # 归一化角度到[-π, π]
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        # 更新状态
        self.state = np.array([x, y, yaw, velocity, steering])

        # 记录轨迹
        self.trajectory.append((x, y, yaw))

    def get_rear_pose(self):
        """获取后轴中心位置和朝向"""
        x, y, yaw, _, _ = self.state
        # 后轴中心在车辆中心后方wheelbase/2处
        rear_x = x - (self.wheelbase / 2) * math.cos(yaw)
        rear_y = y - (self.wheelbase / 2) * math.sin(yaw)
        return rear_x, rear_y, yaw

    def get_vertices(self):
        """获取车辆顶点坐标用于绘制"""
        x, y, yaw, _, _ = self.state
        # 车辆中心为参考点
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 车辆四个角点相对于中心的坐标
        half_length = self.length / 2
        half_width = self.width / 2

        # 未旋转的角点
        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        # 旋转变换矩阵
        # x' = r * cos(alpha + theta)
        # y' = r * sin(alpha + theta)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])

        # 旋转并平移角点
        rotated_corners = corners @ rotation_matrix.T
        vertices = rotated_corners + np.array([x, y])

        # 闭合多边形
        vertices = np.vstack([vertices, vertices[0]])
        return vertices


class PurePursuitController:
    """改进的纯追踪控制器 - 多目标点选择"""

    def __init__(self, speed=SPEED, wheelbase=WHEELBASE, beta=BETA):
        # 控制器参数
        self.target_speed = speed
        self.wheelbase = wheelbase
        self.beta = beta  # 曲率影响因子

        # 公式系数 (需要根据实际情况调整)
        self.A = 0.05
        self.B = 0.1
        self.C = 0.3
        self.M = 1.0

        # 代价函数权重
        self.w1 = 0.1  # 距离权重
        self.w2 = 0.3  # 曲率权重
        self.w3 = 0.4  # 方向权重
        self.w4 = 0.2  # 横向误差权重

        # 缓存上次的目标点索引
        self.last_target_idx = 0
        self.target_point = None
        self.candidate_indices = None
        self.lookahead_distance = 0.0

    # 设置权重参数
    def setW(self, w1, w2, w3, w4):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    # 计算预瞄距离
    def calculate_lookahead(self, velocity, kappa):
        """根据速度和曲率计算预瞄距离"""
        # 改进的预瞄距离公式，考虑速度平方和曲率影响
        # lookahead_distance = (A * v^2 + B * v + C) / (β*|κ| + M)
        lookahead_distance = (self.A * velocity ** 2 + self.B * velocity + self.C) / (
                self.beta * abs(kappa) + self.M)
        # 限制预瞄距离范围
        lookahead_distance = np.clip(lookahead_distance, 0.25, 5.0)
        # if(kappa == 0):
        #     lookahead_distance *= 1.5
        return lookahead_distance

    # 通过前后两点计算中间点的平均曲率
    def calculate_curvature(self, p1, p2, p3):
        """计算三个点形成的曲率"""
        # 计算向量
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # 计算向量长度
        len_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        # 如果长度太小，返回零曲率
        if len_v1 < 1e-5 or len_v2 < 1e-5:
            return 0.0

        # 计算角度变化
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        delta_angle = angle2 - angle1

        # 归一化角度差到[-π, π]
        delta_angle = math.atan2(math.sin(delta_angle), math.cos(delta_angle))

        # 计算平均长度
        avg_length = (len_v1 + len_v2) / 2.0

        # 计算曲率 (角度变化率)
        if avg_length > 1e-5:
            kappa = abs(delta_angle) / avg_length
        else:
            kappa = 0.0

        return kappa

    # 根据预瞄距离寻找目标点
    def find_candidate_targets(self, vehicle_rear_x, vehicle_rear_y, vehicle_yaw, current_speed,
                               waypoints, curvatures):
        """找到候选目标点集合"""
        # 如果路径点为空，返回空列表
        if not waypoints or len(waypoints) < 3:
            return []

        # 计算车辆到每个路径点的距离
        distances = []
        for wp in waypoints:
            dx = wp[0] - vehicle_rear_x
            dy = wp[1] - vehicle_rear_y
            distances.append(math.sqrt(dx ** 2 + dy ** 2))

        # 找到最近的路径点
        closest_idx = np.argmin(distances)

        # 计算最近点处的曲率
        # 确保索引在有效范围内
        idx1 = max(0, closest_idx - 1)
        idx2 = min(len(waypoints) - 1, closest_idx + 1)
        kappa = self.calculate_curvature(waypoints[idx1], waypoints[closest_idx], waypoints[idx2])

        # 根据公式计算预瞄距离
        lookahead_distance = self.calculate_lookahead(current_speed, kappa)
        self.lookahead_distance = lookahead_distance

        # 在预瞄距离附近寻找候选目标点
        candidate_indices = []
        search_window = WINDOW  # 搜索窗口大小

        # 从前一个点开始搜索
        start_idx = max(0, closest_idx - WINDOW)
        end_idx = min(len(waypoints), closest_idx + WINDOW + 1)

        """ change_flag1 """
        for i in range(start_idx, end_idx):
            if distances[i] >= lookahead_distance * 0.8 and distances[i] <= lookahead_distance * 1.5 and i > closest_idx:
                candidate_indices.append(i)

        # 如果没有找到候选点，在更大范围内搜索
        if not candidate_indices:
            for i in range(closest_idx, len(waypoints)):
                if distances[i] >= lookahead_distance:
                    candidate_indices.append(i)
                    break

        # 如果仍然没有候选点，使用原来的单点选择方法
        if not candidate_indices:
            target_idx = closest_idx
            for i in range(closest_idx, len(waypoints)):
                if distances[i] >= lookahead_distance:
                    target_idx = i
                    break
            candidate_indices = [target_idx]

        #  候选点附近的目标点
        candidate_index_final = candidate_indices[0]
        candidate_indices_final = []
        # 像后为2个点的时候还不会有明显的摇摆，但是增加到3个点的时候就会出现明显的摇摆，此时可能真正使用到了代价决策，明显摇摆，控制不好
        for i in range(max(0, candidate_index_final - 3), min(candidate_index_final + 5, len(waypoints))):
            candidate_indices_final.append(i)
        print(candidate_index_final, " target_indices:", candidate_indices_final)

        return candidate_indices_final

    def calculate_cost(self, idx, vehicle_rear_x, vehicle_rear_y, vehicle_yaw, current_speed,
                       waypoints, curvatures, lateral_errors):
        """计算单个候选点的代价"""
        # 获取目标点坐标
        target_x, target_y = waypoints[idx]

        # 计算到目标点的距离
        dx = target_x - vehicle_rear_x
        dy = target_y - vehicle_rear_y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # 计算目标点方向向量
        alpha = math.atan2(dy, dx) - vehicle_yaw
        # 归一化角度到[-π, π]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # 获取目标点曲率
        kappa = curvatures[idx] if idx < len(curvatures) else 0.0

        # 获取横向误差
        lateral_error = lateral_errors[idx] if idx < len(lateral_errors) else 0.0

        # 计算各项代价分量
        # 距离项 (归一化到[0,1])
        d_min, d_max = 0.3, 5.0
        distance_cost = (distance - d_min) / (d_max - d_min) if (d_max - d_min) > 1e-5 else 0.0

        # 曲率项 (归一化到[0,1])
        kappa_d = abs(kappa) * distance
        kappa_d_min, kappa_d_max = 0.0, 2.5  # 需要根据实际数据调整
        if (kappa_d_max - kappa_d_min) > 1e-5:
            curvature_cost = (kappa_d - kappa_d_min) / (kappa_d_max - kappa_d_min)
        else:
            curvature_cost = 0.0

        # 方向项 (归一化到[0,1])
        alpha_max = math.pi
        direction_cost = abs(alpha) / alpha_max if alpha_max > 1e-5 else 0.0

        # 横向误差项 (归一化到[0,1])
        e_y_max = 0.5  # 最大允许横向误差
        lateral_error_cost = abs(lateral_error) / e_y_max if e_y_max > 1e-5 else 0.0

        # 计算总代价
        total_cost = (self.w1 * distance_cost +
                      self.w2 * curvature_cost +
                      self.w3 * direction_cost +
                      self.w4 * lateral_error_cost)

        return total_cost

    def find_target_index(self, vehicle_rear_x, vehicle_rear_y, vehicle_yaw, current_speed,
                          waypoints, curvatures, lateral_errors):
        """使用多目标点选择策略找到最佳目标点"""
        # 获取候选目标点
        candidate_indices = self.find_candidate_targets(vehicle_rear_x, vehicle_rear_y, vehicle_yaw,
                                                        current_speed, waypoints, curvatures)
        self.candidate_indices = candidate_indices

        # 如果只有一个候选点，直接返回
        if len(candidate_indices) <= 1:
            target_idx = candidate_indices[0] if candidate_indices else 0
            # 如果到达终点，从头开始
            if target_idx >= len(waypoints) - 6:
                target_idx = len(waypoints) - 1
            self.last_target_idx = target_idx
            return target_idx

        # 计算每个候选点的代价
        costs = []
        for idx in candidate_indices:
            cost = self.calculate_cost(idx, vehicle_rear_x, vehicle_rear_y, vehicle_yaw,
                                       current_speed, waypoints, curvatures, lateral_errors)
            costs.append(cost)

        # 选择代价最小的点
        min_cost_idx = np.argmin(costs)
        #print("candidate_indices: ", candidate_indices, "  min_cost_index: ", min_cost_idx)
        target_idx = candidate_indices[min_cost_idx]

        # 如果到达终点，从头开始
        if target_idx >= len(waypoints) - 6:
            target_idx = len(waypoints) - 1

        # 记录本次目标索引
        self.last_target_idx = target_idx

        return target_idx

    def calculate_steering(self, vehicle_rear_x, vehicle_rear_y, vehicle_yaw, target_x, target_y):
        """计算转向角度"""
        # 计算目标点方向向量
        dx = target_x - vehicle_rear_x
        dy = target_y - vehicle_rear_y

        # 计算目标点在车辆坐标系中的角度
        alpha = math.atan2(dy, dx) - vehicle_yaw
        # 归一化角度到[-π, π]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # 计算到目标点的距离
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # 纯追踪公式计算转向角度
        if distance > 0.01:
            steering_angle = math.atan(2 * self.wheelbase * math.sin(alpha) / distance)
        else:
            steering_angle = 0.0

        # 限制转向角度范围
        max_steer = MAX_STEER  # 约34度
        steering_angle = np.clip(steering_angle, -max_steer, max_steer)

        return steering_angle

    def get_control(self, vehicle_rear_x, vehicle_rear_y, vehicle_yaw, current_speed,
                    waypoints, curvatures, lateral_errors):
        """获取控制指令"""
        # 找到目标路径点
        target_idx = self.find_target_index(vehicle_rear_x, vehicle_rear_y, vehicle_yaw,
                                            current_speed, waypoints, curvatures, lateral_errors)
        target_x, target_y = waypoints[target_idx]
        self.target_point = (target_x, target_y)

        # 计算转向角度
        steering_angle = self.calculate_steering(
            vehicle_rear_x, vehicle_rear_y, vehicle_yaw, target_x, target_y)

        return self.target_speed, steering_angle


class WaypointLoader:
    """路径点加载器"""

    def __init__(self, file_path, velocity=SPEED):
        self.waypoints = []
        self.curvatures = []  # 存储每个点的曲率
        self.load_waypoints(file_path, velocity)
        if self.waypoints:
            self.waypoint_tree = KDTree(self.waypoints)
        else:
            self.waypoint_tree = None

    def load_waypoints(self, file_path, velocity):
        """从CSV文件加载路径点"""
        try:
            with open(file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    if len(row) >= 2:
                        try:
                            x = float(row[0])
                            y = float(row[1])
                            self.waypoints.append((x, y))
                        except ValueError:
                            continue
            print(f"成功加载 {len(self.waypoints)} 个路径点")

            # 计算路径曲率
            self.calculate_path_curvature()

            if len(self.waypoints) == 0:
                print("警告: 路径文件为空，创建默认路径")
                self.create_complex_path(points=500)
        except Exception as e:
            print(f"加载路径点失败: {e}")
            # 创建复杂路径
            self.create_complex_path(points=500)

    def calculate_path_curvature(self):
        """计算路径上每个点的曲率"""
        self.curvatures = []
        if len(self.waypoints) < 3:
            # 如果点太少，曲率设为0
            self.curvatures = [0.0] * len(self.waypoints)
            return

        # 使用三点法计算曲率
        for i in range(len(self.waypoints)):
            if i == 0:
                # 第一个点使用第二个点的曲率
                kappa = self.calculate_curvature(self.waypoints[0], self.waypoints[1], self.waypoints[2])
            elif i == len(self.waypoints) - 1:
                # 最后一个点使用倒数第二个点的曲率
                kappa = self.curvatures[-1]
            else:
                kappa = self.calculate_curvature(self.waypoints[i - 1], self.waypoints[i], self.waypoints[i + 1])
            self.curvatures.append(kappa)

    def calculate_curvature(self, p1, p2, p3):
        """计算三个点形成的曲率"""
        # 计算向量
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # 计算向量长度
        len_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        # 如果长度太小，返回零曲率
        if len_v1 < 1e-5 or len_v2 < 1e-5:
            return 0.0

        # 计算角度变化
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        delta_angle = angle2 - angle1

        # 归一化角度差到[-π, π]
        delta_angle = math.atan2(math.sin(delta_angle), math.cos(delta_angle))

        # 计算平均长度
        avg_length = (len_v1 + len_v2) / 2.0

        # 计算曲率 (角度变化率)
        if avg_length > 1e-5:
            kappa = abs(delta_angle) / avg_length
        else:
            kappa = 0.0

        return kappa

    def create_complex_path(self, points=1000):
        """根据PATH_TYPE创建不同类型的路径"""
        if PATH_TYPE == 0:
            # 创建具有连续变化曲率的光滑路径，用于验证预瞄距离中曲率变化的影响
            self.create_continuous_curvature_path(points)
        elif PATH_TYPE == 1:
            # 创建直线+圆弧组合路径：长直线6m+四分之一圆弧（R为0.6m）+直线4m + 四分之一圆弧（与上一个圆弧对称，R为0.6m）+ 长直线6m
            self.create_straight_arc_path(points)
        else:
            # 默认使用连续变化曲率路径
            self.create_continuous_curvature_path(points)

    def create_continuous_curvature_path(self, points=1000):
        """创建具有连续变化曲率的光滑路径，用于验证预瞄距离中曲率变化的影响"""
        self.waypoints = []

        # 总长度约20米
        total_length = 20.0

        # 使用积分法生成曲率连续变化的路径
        # 定义曲率函数kappa(s)，其中s是路径长度参数
        def curvature_function(s_normalized):
            """
            定义曲率函数，s_normalized是归一化的路径参数[0,1]
            这个函数会产生不规则但连续变化的曲率
            """
            # 组合多个频率的三角函数产生复杂的曲率变化
            kappa = (1.0 * math.sin(2 * math.pi * s_normalized * 2) +
                     0.6 * math.sin(2 * math.pi * s_normalized * 5 + 0.5) +
                     0.4 * math.sin(2 * math.pi * s_normalized * 10 + 1.2) +
                     0.2 * math.sin(2 * math.pi * s_normalized * 15 + 2.0))
            return kappa

        # 数值积分生成路径点
        # 初始条件
        x, y = 0.0, 0.0
        theta = 0.0  # 初始方向角
        ds = total_length / (points - 1)  # 路径步长

        # 添加初始点
        self.waypoints.append((x, y))

        # 逐步积分生成路径点
        for i in range(1, points):
            s_normalized = i / (points - 1)

            # 获取当前点的曲率
            kappa = curvature_function(s_normalized)

            # 更新方向角 (theta += kappa * ds)
            theta += kappa * ds

            # 更新位置
            x += math.cos(theta) * ds
            y += math.sin(theta) * ds

            # 添加路径点
            self.waypoints.append((x, y))

        print(f"创建了具有连续变化曲率的路径，包含 {len(self.waypoints)} 个点")

        # 计算路径曲率
        self.calculate_path_curvature()

        # 保存路径到文件
        try:
            with open("others/complex_path.csv", 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for wp in self.waypoints:
                    csvwriter.writerow([wp[0], wp[1]])
            print("复杂路径已保存到 complex_path.csv")
        except Exception as e:
            print(f"保存路径失败: {e}")

    def create_straight_arc_path(self, points=1000):
        """创建直线+圆弧组合路径：长直线6m+四分之一圆弧（R为0.6m）+直线4m + 四分之一圆弧（与上一个圆弧对称，R为0.6m）+ 长直线6m"""
        self.waypoints = []

        # 定义路径参数
        straight1_length = 6.0  # 第一段直线长度
        arc_radius = 0.6        # 圆弧半径
        arc_angle = math.pi / 2 # 四分之一圆弧角度 (90度)
        straight2_length = 4.0  # 中间直线长度
        straight3_length = 6.0  # 第三段直线长度

        # 计算各段路径的长度
        arc_length = arc_radius * arc_angle  # 圆弧长度
        total_length = straight1_length + arc_length + straight2_length + arc_length + straight3_length

        # 计算各段路径的点数
        straight1_points = int(points * straight1_length / total_length)
        arc1_points = int(points * arc_length / total_length)
        straight2_points = int(points * straight2_length / total_length)
        arc2_points = int(points * arc_length / total_length)
        straight3_points = points - straight1_points - arc1_points - straight2_points - arc2_points  # 剩余点数分配给第三段直线

        # 生成第一段直线 (沿X轴正方向)
        for i in range(straight1_points):
            x = (i / straight1_points) * straight1_length
            y = 0.0
            self.waypoints.append((x, y))

        # 生成第一段圆弧 (顺时针，从X轴正方向到Y轴负方向)
        start_angle = math.pi / 2  # 初始角度
        center_x = straight1_length  # 圆心X坐标
        center_y = -arc_radius      # 圆心Y坐标
        for i in range(arc1_points):
            angle = start_angle - (i / arc1_points) * arc_angle  # 顺时针方向
            x = center_x + arc_radius * math.cos(angle)
            y = center_y + arc_radius * math.sin(angle)
            self.waypoints.append((x, y))

        # 生成第二段直线 (沿Y轴负方向)
        start_y = center_y  # 从圆弧终点开始
        start_x = center_x + arc_radius
        for i in range(straight2_points):
            y = start_y - (i / straight2_points) * straight2_length
            x = start_x
            self.waypoints.append((x, y))

        # 生成第二段圆弧 (逆时针，从Y轴负方向回到X轴正方向)
        center_x = start_x - arc_radius  # 新圆弧的圆心X坐标
        center_y = y  # 新圆弧的圆心Y坐标
        start_angle = 0  # 从Y轴负方向开始
        for i in range(arc2_points):
            angle = start_angle - (i / arc2_points) * arc_angle  # 逆时针方向
            x = center_x + arc_radius * math.cos(angle)
            y = center_y + arc_radius * math.sin(angle)
            self.waypoints.append((x, y))

        # 生成第三段直线 (沿X轴正方向)
        start_x = center_x # 从圆弧终点开始
        start_y = center_y - arc_radius
        for i in range(straight3_points):
            x = start_x - (i / straight3_points) * straight3_length
            y = start_y
            self.waypoints.append((x, y))

        print(f"创建了直线+圆弧组合路径，包含 {len(self.waypoints)} 个点")
        print(f"路径组成: 直线{straight1_length}m + 圆弧(R={arc_radius}m) + 直线{straight2_length}m + 圆弧(R={arc_radius}m) + 直线{straight3_length}m")

        # 计算路径曲率
        self.calculate_path_curvature()

        # 保存路径到文件
        try:
            with open("others/complex_path.csv", 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for wp in self.waypoints:
                    csvwriter.writerow([wp[0], wp[1]])
            print("直线+圆弧组合路径已保存到 complex_path.csv")
        except Exception as e:
            print(f"保存路径失败: {e}")

class Simulation:
    """路径跟踪仿真系统"""
    def __init__(self):

        self.reached_end = False  # 到达终点标志
        self.simulation_finalized = False  # 仿真结束标志

        # 创建车辆
        self.vehicle = AckermannVehicle()

        # 加载路径点
        self.waypoint_loader = WaypointLoader("others/complex_path.csv")
        self.waypoints = self.waypoint_loader.waypoints
        self.path_curvatures = self.waypoint_loader.curvatures

        # 创建控制器
        self.controller = PurePursuitController()

        # 设置可视化
        self.fig = plt.figure(figsize=(16, 16))  # 增加高度以适应更多图表
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.5, wspace=0.4)  # 增加行间距

        # 主路径图
        self.ax_main = self.fig.add_subplot(gs[0:2, 0])  # 占据前两行所有列
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True)
        self.ax_main.set_title("阿克曼小车路径跟踪仿真 - 多目标点选择")
        self.ax_main.set_xlabel("X (m)")
        self.ax_main.set_ylabel("Y (m)")

        # 绘制路径
        if self.waypoints:
            wp_x, wp_y = zip(*self.waypoints)
            self.path_line, = self.ax_main.plot(wp_x, wp_y, 'b-', alpha=0.5, label="参考路径")

            # 绘制曲率图
            curvature_colors = []
            for k in self.path_curvatures:
                # 根据曲率大小设置颜色
                if k < 0.05:
                    color = 'green'  # 低曲率 - 直线
                elif k < 0.2:
                    color = 'blue'  # 中等曲率
                else:
                    color = 'yellow'  # 高曲率
                curvature_colors.append(color)

            # 绘制带有曲率颜色的路径
            for i in range(len(wp_x) - 1):
                self.ax_main.plot([wp_x[i], wp_x[i + 1]], [wp_y[i], wp_y[i + 1]],
                                  color=curvature_colors[i], linewidth=2, alpha=0.7)

        # 绘制车辆
        self.vehicle_outline, = self.ax_main.plot([], [], 'k-', linewidth=2)
        self.front_wheels, = self.ax_main.plot([], [], 'r-', linewidth=2)
        self.rear_axle, = self.ax_main.plot([], [], 'go', markersize=8)

        # 绘制轨迹
        self.trajectory_line, = self.ax_main.plot([], [], 'r-', alpha=0.6, label="实际轨迹")

        # 绘制目标点
        self.target_point, = self.ax_main.plot([], [], 'ro', markersize=8, label="目标点")

        # 绘制候选目标点
        self.candidate_points, = self.ax_main.plot([], [], 'mo', markersize=4, label="候选目标点")

        # 绘制预瞄距离圆
        self.lookahead_circle = plt.Circle((0, 0), 0, color='r', fill=False, linestyle='--', alpha=0.5)
        self.ax_main.add_patch(self.lookahead_circle)

        # 添加图例
        self.ax_main.legend(loc='upper right')

        # 设置坐标范围
        if self.waypoints:
            min_x, min_y = np.min(self.waypoints, axis=0)
            max_x, max_y = np.max(self.waypoints, axis=0)
            margin = 3.0
            self.ax_main.set_xlim(min_x - margin, max_x + margin)
            self.ax_main.set_ylim(min_y - margin, max_y + margin)
        else:
            self.ax_main.set_xlim(-3, 3)
            self.ax_main.set_ylim(-3, 3)

        # 添加文本状态显示
        self.status_text = self.ax_main.text(0.02, 0.95, '', transform=self.ax_main.transAxes)

        # 局部放大图
        self.ax_zoom = self.fig.add_subplot(gs[0:2, 1])
        self.ax_zoom.set_title("车辆位置局部放大图")
        self.ax_zoom.set_xlabel("X (m)")
        self.ax_zoom.set_ylabel("Y (m)")
        self.ax_zoom.grid(True)
        self.ax_zoom.set_aspect('equal')

        # 局部放大图的元素
        self.zoom_path_line, = self.ax_zoom.plot([], [], 'b-', alpha=0.7)
        self.zoom_vehicle_outline, = self.ax_zoom.plot([], [], 'k-', linewidth=2)
        self.zoom_front_wheels, = self.ax_zoom.plot([], [], 'r-', linewidth=2)
        self.zoom_rear_axle, = self.ax_zoom.plot([], [], 'go', markersize=6)
        self.zoom_trajectory_line, = self.ax_zoom.plot([], [], 'g-', alpha=0.6)
        self.zoom_target_point, = self.ax_zoom.plot([], [], 'ro', markersize=6)
        self.zoom_candidate_points, = self.ax_zoom.plot([], [], 'mo', markersize=4)
        self.zoom_lookahead_circle = plt.Circle((0, 0), 0, color='r', fill=False, linestyle='--', alpha=0.5)
        self.ax_zoom.add_patch(self.zoom_lookahead_circle)

        # ====== 第二行图表 ======
        # 横向误差图
        self.ax_error = self.fig.add_subplot(gs[2, 1])
        self.ax_error.set_title("横向误差")
        self.ax_error.set_xlabel("时间步")
        self.ax_error.set_ylabel("误差 (m)")
        self.error_line, = self.ax_error.plot([], [], 'r-')
        self.ax_error.grid(True)

        # 预瞄距离图
        self.ax_lookahead = self.fig.add_subplot(gs[2, 2])
        self.ax_lookahead.set_title("预瞄距离变化")
        self.ax_lookahead.set_xlabel("时间步")
        self.ax_lookahead.set_ylabel("距离 (m)")
        self.lookahead_line, = self.ax_lookahead.plot([], [], 'b-')
        self.ax_lookahead.grid(True)

        # ====== 第三行图表 ======
        # 转向角图
        self.ax_steering = self.fig.add_subplot(gs[3, 0])
        self.ax_steering.set_title("转向角变化")
        self.ax_steering.set_xlabel("时间步", labelpad=10)  # 增加标签间距
        self.ax_steering.set_ylabel("角度 (度)")
        self.steering_line, = self.ax_steering.plot([], [], 'g-')
        self.ax_steering.grid(True)

        # 速度图
        self.ax_speed = self.fig.add_subplot(gs[3, 1])
        self.ax_speed.set_title("速度变化")
        self.ax_speed.set_xlabel("时间步", labelpad=10)  # 增加标签间距
        self.ax_speed.set_ylabel("速度 (m/s)")
        self.speed_line, = self.ax_speed.plot([], [], 'm-')
        self.ax_speed.grid(True)

        # 曲率图
        self.ax_curvature = self.fig.add_subplot(gs[3, 2])
        self.ax_curvature.set_title("路径曲率变化")
        self.ax_curvature.set_xlabel("时间步", labelpad=10)  # 增加标签间距
        self.ax_curvature.set_ylabel("曲率")
        self.curvature_line, = self.ax_curvature.plot([], [], 'c-')
        self.ax_curvature.grid(True)

        # 设置动画对象
        self.ani = None
        self.time_steps = 0
        self.max_steps = 4000  # 最大仿真步数

        # 性能指标
        self.lateral_errors = []
        self.lookahead_distances = []
        self.steering_angles = []
        self.speeds = []
        self.current_curvatures = []  # 记录当前路径点的曲率

        # 创建输出目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"result/simulation_results_{timestamp}_beta_{self.controller.beta}_multi_target"
        # 检查是否需要保存结果文件到result文件夹
        if SAVE_RESULT_FILES:
            os.makedirs(self.output_dir, exist_ok=True)

    def init_animation(self):
        """初始化动画元素"""
        # 主图元素
        self.vehicle_outline.set_data([], [])
        self.front_wheels.set_data([], [])
        self.rear_axle.set_data([], [])
        self.trajectory_line.set_data([], [])
        self.target_point.set_data([], [])
        self.candidate_points.set_data([], [])
        self.lookahead_circle.center = (0, 0)
        self.lookahead_circle.set_radius(0)
        self.status_text.set_text('')

        # 局部放大图元素
        self.zoom_path_line.set_data([], [])
        self.zoom_vehicle_outline.set_data([], [])
        self.zoom_front_wheels.set_data([], [])
        self.zoom_rear_axle.set_data([], [])
        self.zoom_trajectory_line.set_data([], [])
        self.zoom_target_point.set_data([], [])
        self.zoom_candidate_points.set_data([], [])
        self.zoom_lookahead_circle.center = (0, 0)
        self.zoom_lookahead_circle.set_radius(0)

        # 性能图表元素
        self.error_line.set_data([], [])
        self.lookahead_line.set_data([], [])
        self.steering_line.set_data([], [])
        self.speed_line.set_data([], [])
        self.curvature_line.set_data([], [])

        return (self.vehicle_outline, self.front_wheels, self.rear_axle,
                self.trajectory_line, self.target_point, self.candidate_points, self.lookahead_circle,
                self.status_text, self.zoom_path_line, self.zoom_vehicle_outline,
                self.zoom_front_wheels, self.zoom_rear_axle, self.zoom_trajectory_line,
                self.zoom_target_point, self.zoom_candidate_points, self.zoom_lookahead_circle, self.error_line,
                self.lookahead_line, self.steering_line, self.speed_line, self.curvature_line)

    def update_zoom_view(self, vehicle_x, vehicle_y):
        """更新局部放大图"""
        # 设置局部放大范围（车辆周围5米）
        zoom_radius = 0.5
        self.ax_zoom.set_xlim(vehicle_x - zoom_radius, vehicle_x + zoom_radius)
        self.ax_zoom.set_ylim(vehicle_y - zoom_radius, vehicle_y + zoom_radius)

        # 更新局部放大图的路径
        if self.waypoints:
            # 只显示车辆附近的路径点
            zoom_path_x = []
            zoom_path_y = []
            for wp in self.waypoints:
                if abs(wp[0] - vehicle_x) < zoom_radius * 2 and abs(wp[1] - vehicle_y) < zoom_radius * 2:
                    zoom_path_x.append(wp[0])
                    zoom_path_y.append(wp[1])
            self.zoom_path_line.set_data(zoom_path_x, zoom_path_y)

    def calculate_lateral_error(self, vehicle_x, vehicle_y, vehicle_yaw):
        """计算横向误差"""
        if not self.waypoints or not hasattr(self, 'waypoint_tree'):
            return 0.0

        # 找到最近的路径点
        dist, idx = self.waypoint_tree.query([vehicle_x, vehicle_y])
        path_x, path_y = self.waypoints[idx]

        # 记录当前路径点的曲率
        if idx < len(self.path_curvatures):
            self.current_curvatures.append(self.path_curvatures[idx])
        else:
            self.current_curvatures.append(0.0)

        # 计算路径点处的切向量
        if idx < len(self.waypoints) - 1:
            next_x, next_y = self.waypoints[idx + 1]
            tangent_vector = (next_x - path_x, next_y - path_y)
        elif idx > 0:
            prev_x, prev_y = self.waypoints[idx - 1]
            tangent_vector = (path_x - prev_x, path_y - prev_y)
        else:
            tangent_vector = (1, 0)  # 默认方向

        # 计算切向量的角度
        tangent_angle = math.atan2(tangent_vector[1], tangent_vector[0])

        # 计算车辆到路径点的向量
        dx = vehicle_x - path_x
        dy = vehicle_y - path_y

        # 计算横向误差（法向分量）
        lateral_error = -dx * math.sin(tangent_angle) + dy * math.cos(tangent_angle)

        return lateral_error

    def update(self, frame):
        """更新仿真状态"""
        # 如果已到达终点，直接返回
        if self.reached_end:
            return (self.vehicle_outline, self.front_wheels, self.rear_axle,
                    self.trajectory_line, self.target_point, self.candidate_points, self.lookahead_circle,
                    self.status_text, self.zoom_path_line, self.zoom_vehicle_outline,
                    self.zoom_front_wheels, self.zoom_rear_axle, self.zoom_trajectory_line,
                    self.zoom_target_point, self.zoom_candidate_points, self.zoom_lookahead_circle, self.error_line,
                    self.lookahead_line, self.steering_line, self.speed_line, self.curvature_line)

        if self.time_steps >= self.max_steps:
            # 仿真结束
            self.finalize_simulation()
            return (self.vehicle_outline, self.front_wheels, self.rear_axle,
                    self.trajectory_line, self.target_point, self.candidate_points, self.lookahead_circle,
                    self.status_text, self.zoom_path_line, self.zoom_vehicle_outline,
                    self.zoom_front_wheels, self.zoom_rear_axle, self.zoom_trajectory_line,
                    self.zoom_target_point, self.zoom_candidate_points, self.zoom_lookahead_circle, self.error_line,
                    self.lookahead_line, self.steering_line, self.speed_line, self.curvature_line)

        # 获取车辆后轴中心位置
        rear_x, rear_y, rear_yaw = self.vehicle.get_rear_pose()

        # 检查是否到达终点
        if self.waypoints and not self.reached_end:
            # 获取最后一个路径点
            last_wp = self.waypoints[-1]
            # 计算到终点的距离
            dist_to_end = math.sqrt((rear_x - last_wp[0]) ** 2 + (rear_y - last_wp[1]) ** 2)

            # 如果距离小于阈值，标记到达终点并停止车辆
            if dist_to_end < 1:  # 0.1米阈值
                self.reached_end = True
                # 停止车辆
                self.vehicle.update(0, 0)  # 速度设为0，转向角设为0
                # 结束仿真
                self.finalize_simulation()
                return (self.vehicle_outline, self.front_wheels, self.rear_axle,
                        self.trajectory_line, self.target_point, self.candidate_points, self.lookahead_circle,
                        self.status_text, self.zoom_path_line, self.zoom_vehicle_outline,
                        self.zoom_front_wheels, self.zoom_rear_axle, self.zoom_trajectory_line,
                        self.zoom_target_point, self.zoom_candidate_points, self.zoom_lookahead_circle, self.error_line,
                        self.lookahead_line, self.steering_line, self.speed_line, self.curvature_line)

        # 获取车辆当前速度
        current_speed = self.vehicle.state[3]

        # 计算横向误差并记录
        lateral_error = self.calculate_lateral_error(rear_x, rear_y, rear_yaw)
        self.lateral_errors.append(lateral_error)

        # 获取控制指令
        if self.waypoints and not self.reached_end:  # 添加未到达终点的条件            # 计算所有路径点的横向误差（用于代价函数）
            # 修复：只计算有限数量的路径点，避免性能问题
            all_lateral_errors = [0.0] * len(self.waypoints)  # 初始化为0

            # 只计算当前车辆附近一定范围内的路径点的横向误差
            if hasattr(self, 'waypoint_tree'):
                # 查找车辆附近的路径点
                distances, indices = self.waypoint_tree.query([rear_x, rear_y], k=min(20, len(self.waypoints)))
                if np.isscalar(indices):  # 如果只有一个点
                    indices = [indices]

                # 计算这些附近点的横向误差
                for idx in indices:
                    wp_x, wp_y = self.waypoints[idx]
                    error = self.calculate_lateral_error(wp_x, wp_y, rear_yaw)
                    all_lateral_errors[idx] = error

            speed, steering_angle = self.controller.get_control(
                rear_x, rear_y, rear_yaw, current_speed, self.waypoints,
                self.path_curvatures, all_lateral_errors)
        else:
            speed, steering_angle = 0.0, 0.0

        # 记录预瞄距离和转向角
        self.lookahead_distances.append(self.controller.lookahead_distance)
        self.steering_angles.append(steering_angle)
        self.speeds.append(speed)

        # 更新车辆状态（仅当未到达终点）
        if not self.reached_end:
            self.vehicle.update(speed, steering_angle)

        # 更新可视化
        self.update_visualization()

        # 更新局部放大图
        self.update_zoom_view(rear_x, rear_y)

        self.time_steps += 1
        return (self.vehicle_outline, self.front_wheels, self.rear_axle,
                self.trajectory_line, self.target_point, self.candidate_points, self.lookahead_circle,
                self.status_text, self.zoom_path_line, self.zoom_vehicle_outline,
                self.zoom_front_wheels, self.zoom_rear_axle, self.zoom_trajectory_line,
                self.zoom_target_point, self.zoom_candidate_points, self.zoom_lookahead_circle, self.error_line,
                self.lookahead_line, self.steering_line, self.speed_line, self.curvature_line)

    def update_visualization(self):
        """更新可视化元素"""
        # 获取车辆状态
        x, y, yaw, speed, steering_angle = self.vehicle.state
        rear_x, rear_y, _ = self.vehicle.get_rear_pose()

        # 根据SHOW_ANIMATION选项决定是否更新可视化元素
        if SHOW_ANIMATION:
            # ====== 更新主图 ======
            # 更新车辆轮廓
            vertices = self.vehicle.get_vertices()
            self.vehicle_outline.set_data(vertices[:, 0], vertices[:, 1])

            # 更新前轮指示
            wheel_length = 0.1
            front_wheel_x = [x + 0.1 * math.cos(yaw),
                             x + 0.1 * math.cos(yaw) + wheel_length * math.cos(yaw + steering_angle)]
            front_wheel_y = [y + 0.1 * math.sin(yaw),
                             y + 0.1 * math.sin(yaw) + wheel_length * math.sin(yaw + steering_angle)]
            self.front_wheels.set_data(front_wheel_x, front_wheel_y)

            # 更新后轴中心位置
            self.rear_axle.set_data([rear_x], [rear_y])

            # 更新轨迹
            if self.vehicle.trajectory:
                traj_x, traj_y, _ = zip(*self.vehicle.trajectory)
                self.trajectory_line.set_data(traj_x, traj_y)

            # 更新目标点和候选点
            if self.controller.target_point:
                target_x, target_y = self.controller.target_point
                self.target_point.set_data([target_x], [target_y])

            # 更新候选目标点（在主图中显示）
            # 这里简化处理，只显示当前选择的目标点周围的几个点
            candidate_x = []
            candidate_y = []
            # if hasattr(self.controller, 'last_target_idx') and self.waypoints:
            #     base_idx = self.controller.last_target_idx
            #     for i in range(max(0, base_idx - 3), min(len(self.waypoints), base_idx + 4)):
            #         candidate_x.append(self.waypoints[i][0])
            #         candidate_y.append(self.waypoints[i][1])

            # 按照原来的候选点全部显示
            if hasattr(self.controller, 'last_target_idx') and self.waypoints:
                base_idx = self.controller.last_target_idx
                for i in range(self.controller.candidate_indices[0], self.controller.candidate_indices[-1]):
                    candidate_x.append(self.waypoints[i][0])
                    candidate_y.append(self.waypoints[i][1])
            self.candidate_points.set_data(candidate_x, candidate_y)

            # 更新预瞄距离圆
            self.lookahead_circle.center = (rear_x, rear_y)
            self.lookahead_circle.set_radius(self.controller.lookahead_distance)

            # 更新状态文本
            status_str = ""
            if self.reached_end:
                status_str = "已到达终点！\n"

            status_str += (f"位置: ({x:.2f}, {y:.2f})\n"
                           f"朝向: {math.degrees(yaw):.1f}°\n"
                           f"速度: {speed:.2f} m/s\n"
                           f"转向角: {math.degrees(steering_angle):.1f}°\n"
                           f"预瞄距离: {self.controller.lookahead_distance:.2f} m\n"
                           f"路径曲率: {self.path_curvatures[min(self.controller.last_target_idx, len(self.path_curvatures) - 1)]:.4f}\n"
                           f"目标点索引: {self.controller.last_target_idx}/{len(self.waypoints) - 1}\n"
                           f"时间步: {self.time_steps}/{self.max_steps}")
            self.status_text.set_text(status_str)

            # ====== 更新局部放大图 ======
            # 更新车辆轮廓
            self.zoom_vehicle_outline.set_data(vertices[:, 0], vertices[:, 1])

            # 更新前轮指示
            self.zoom_front_wheels.set_data(front_wheel_x, front_wheel_y)

            # 更新后轴中心位置
            self.zoom_rear_axle.set_data([rear_x], [rear_y])

            # 更新轨迹
            if self.vehicle.trajectory:
                # 只显示最近的轨迹点
                recent_traj = self.vehicle.trajectory[-50:] if len(
                    self.vehicle.trajectory) > 50 else self.vehicle.trajectory
                traj_x, traj_y, _ = zip(*recent_traj)
                self.zoom_trajectory_line.set_data(traj_x, traj_y)

            # 更新目标点
            if self.controller.target_point:
                self.zoom_target_point.set_data([self.controller.target_point[0]], [self.controller.target_point[1]])

            # 更新候选点
            self.zoom_candidate_points.set_data(candidate_x, candidate_y)

            # 更新预瞄距离圆
            self.zoom_lookahead_circle.center = (rear_x, rear_y)
            self.zoom_lookahead_circle.set_radius(self.controller.lookahead_distance)

        # 根据SHOW_REALTIME_PLOTS选项决定是否更新性能图表
        if SHOW_REALTIME_PLOTS:
            # ====== 更新性能图表 ======
            time_steps = list(range(len(self.lateral_errors)))

            # 横向误差
            self.error_line.set_data(time_steps, self.lateral_errors)
            self.ax_error.relim()
            self.ax_error.autoscale_view()

            # 预瞄距离
            self.lookahead_line.set_data(time_steps, self.lookahead_distances)
            self.ax_lookahead.relim()
            self.ax_lookahead.autoscale_view()

            # 转向角
            self.steering_line.set_data(time_steps, np.degrees(self.steering_angles))
            self.ax_steering.relim()
            self.ax_steering.autoscale_view()

            # 速度
            self.speed_line.set_data(time_steps, self.speeds)
            self.ax_speed.relim()
            self.ax_speed.autoscale_view()

            # 曲率
            if self.current_curvatures:
                # 确保曲率数据与时间步长度一致
                curvature_length = len(self.current_curvatures)
                if curvature_length <= len(time_steps):
                    curvature_time_steps = time_steps[:curvature_length]
                else:
                    # 如果曲率数据更多，则截断
                    curvature_time_steps = time_steps
                    self.current_curvatures = self.current_curvatures[:len(time_steps)]

                self.curvature_line.set_data(curvature_time_steps, self.current_curvatures)
                self.ax_curvature.relim()
                self.ax_curvature.autoscale_view()

    # 修改 finalize_simulation 方法，确保动画停止
    def finalize_simulation(self):
        """仿真结束，计算并输出性能指标"""
        # 检查是否已经执行过finalize_simulation
        if self.simulation_finalized:
            return

        # 标记已完成finalize_simulation
        self.simulation_finalized = True

        # 停止动画
        if self.ani and self.ani.event_source:
            self.ani.event_source.stop()

        # 计算性能指标
        abs_errors = np.abs(self.lateral_errors)
        avg_error = np.mean(abs_errors)
        max_error = np.max(abs_errors)
        error_variance = np.var(self.lateral_errors)
        avg_lookahead = np.mean(self.lookahead_distances)
        avg_speed = np.mean(self.speeds)

        # 计算总行驶路程
        total_distance = 0.0
        if self.vehicle.trajectory:
            for i in range(1, len(self.vehicle.trajectory)):
                x1, y1, _ = self.vehicle.trajectory[i - 1]
                x2, y2, _ = self.vehicle.trajectory[i]
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                total_distance += distance

        # 计算转向角标准差和横向误差标准差
        steering_std = np.std(self.steering_angles) if self.steering_angles else 0.0
        lateral_error_std = np.std(self.lateral_errors) if self.lateral_errors else 0.0

        # 打印性能指标
        print("\n" + "=" * 50)
        print("仿真性能指标:")
        print(f"平均横向误差: {avg_error:.4f} m")
        print(f"最大横向误差: {max_error:.4f} m")
        print(f"横向误差方差: {error_variance:.6f}")
        print(f"横向误差标准差: {lateral_error_std:.6f}")
        print(f"平均预瞄距离: {avg_lookahead:.4f} m")
        print(f"平均速度: {avg_speed:.4f} m/s")
        print(f"转向角标准差: {steering_std:.6f}")
        print(f"总行驶路程: {total_distance:.4f} m")
        print("=" * 50 + "\n")

        # 保存性能指标到txt文件
        if SAVE_RESULT_FILES:
            with open(f"{self.output_dir}/performance_metrics.txt", 'w', encoding='utf-8') as f:
                # 添加UTF-8 BOM，确保正确显示中文字符
                f.write('\ufeff')
                f.write("仿真性能指标:\n")
                f.write(f"平均横向误差: {avg_error:.4f} m\n")
                f.write(f"最大横向误差: {max_error:.4f} m\n")
                f.write(f"横向误差方差: {error_variance:.6f}\n")
                f.write(f"横向误差标准差: {lateral_error_std:.6f}\n")
                f.write(f"平均预瞄距离: {avg_lookahead:.4f} m\n")
                f.write(f"平均速度: {avg_speed:.4f} m/s\n")
                f.write(f"转向角标准差: {steering_std:.6f}\n")
                f.write(f"总行驶路程: {total_distance:.4f} m\n")
                f.write(f"仿真时间步数: {self.time_steps}\n")
                f.write(f"数据点数量: {len(self.lateral_errors)}\n")

        # 将数据输出到CSV文件
        try:
            import csv
            import os

            # 确保输出目录存在（仅当需要保存结果文件时）
            if SAVE_RESULT_FILES:
                os.makedirs(self.output_dir, exist_ok=True)
                csv_file = f"{self.output_dir}/simulation_data.csv"
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    # 添加UTF-8 BOM，确保Excel能正确识别中文字符
                    f.write('\ufeff')
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow(['时刻', 'x', 'y', '行驶路程', '曲率', '转向角', '线速度', '角速度', '预瞄距离', '横向误差'])

                    # 确定数据长度
                    data_length = min(len(self.lateral_errors), len(self.lookahead_distances),
                                      len(self.steering_angles), len(self.speeds),
                                      len(self.current_curvatures), len(self.vehicle.trajectory))

                    # 计算角速度和行驶路程
                    angular_velocities = []
                    distances_traveled = [0.0]  # 初始距离为0

                    for i in range(data_length):
                        # 计算角速度
                        if i < len(self.speeds) and i < len(self.steering_angles):
                            # 角速度 = 线速度 * tan(转向角) / 轴距
                            angular_velocity = self.speeds[i] * math.tan(
                                self.steering_angles[i]) / self.vehicle.wheelbase
                            angular_velocities.append(angular_velocity)
                        else:
                            angular_velocities.append(0.0)

                        # 计算累计行驶路程
                        if i > 0 and i < len(self.vehicle.trajectory):
                            x1, y1, _ = self.vehicle.trajectory[i - 1]
                            x2, y2, _ = self.vehicle.trajectory[i]
                            segment_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            distances_traveled.append(distances_traveled[-1] + segment_distance)
                        elif i == 0:
                            distances_traveled.append(0.0)
                        else:
                            distances_traveled.append(distances_traveled[-1])

                    # 写入数据行
                    for i in range(data_length):
                        # 提取坐标
                        x_coord = self.vehicle.trajectory[i][0] if i < len(self.vehicle.trajectory) else 0.0
                        y_coord = self.vehicle.trajectory[i][1] if i < len(self.vehicle.trajectory) else 0.0

                        writer.writerow([
                            i,
                            x_coord,
                            y_coord,
                            distances_traveled[i] if i < len(distances_traveled) else 0.0,
                            self.current_curvatures[i] if i < len(self.current_curvatures) else 0.0,
                            self.steering_angles[i] if i < len(self.steering_angles) else 0.0,
                            self.speeds[i] if i < len(self.speeds) else 0.0,
                            angular_velocities[i] if i < len(angular_velocities) else 0.0,
                            self.lookahead_distances[i] if i < len(self.lookahead_distances) else 0.0,
                            self.lateral_errors[i] if i < len(self.lateral_errors) else 0.0
                        ])
                print(f"仿真数据已保存到 {csv_file}")
        except Exception as e:
            print(f"保存CSV文件失败: {e}")
            import traceback
            traceback.print_exc()

        # 创建性能图表
        self.create_performance_plots()

        # 将beta参数、平均横向误差、横向误差标准差、转向角标准差追加到res.csv文件末尾
        res_csv_path = "res.csv"
        # 检查文件是否存在，如果不存在则写入表头
        file_exists = os.path.isfile(res_csv_path)
        with open(res_csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['beta', '平均横向误差', '横向误差标准差', '转向角标准差'])
            writer.writerow([self.controller.beta, avg_error, lateral_error_std, steering_std])

    def create_performance_plots(self):
        """创建并保存性能图表"""
        # 如果不保存结果文件，则直接返回
        if not SAVE_RESULT_FILES:
            return

        # 横向误差图
        plt.figure(figsize=(10, 6))
        plt.plot(self.lateral_errors, 'r-')
        plt.title("横向误差变化")
        plt.xlabel("时间步")
        plt.ylabel("横向误差 (m)")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/lateral_error.png")
        plt.close()

        # 预瞄距离图
        plt.figure(figsize=(10, 6))
        plt.plot(self.lookahead_distances, 'b-')
        plt.title("预瞄距离变化")
        plt.xlabel("时间步")
        plt.ylabel("预瞄距离 (m)")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/lookahead_distance.png")
        plt.close()

        # 转向角图
        plt.figure(figsize=(10, 6))
        plt.plot(np.degrees(self.steering_angles), 'g-')
        plt.title("转向角变化")
        plt.xlabel("时间步")
        plt.ylabel("转向角 (度)")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/steering_angle.png")
        plt.close()

        # 速度图
        plt.figure(figsize=(10, 6))
        plt.plot(self.speeds, 'm-')
        plt.title("速度变化")
        plt.xlabel("时间步")
        plt.ylabel("速度 (m/s)")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/speed.png")
        plt.close()

        # 路径曲率图
        plt.figure(figsize=(10, 6))
        plt.plot(self.current_curvatures, 'c-')
        plt.title("车辆位置路径曲率变化")
        plt.xlabel("时间步")
        plt.ylabel("曲率")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/current_curvature.png")
        plt.close()

        # 横向误差分布图
        plt.figure(figsize=(10, 6))
        plt.hist(self.lateral_errors, bins=30, color='orange', alpha=0.7)
        plt.title("横向误差分布")
        plt.xlabel("横向误差 (m)")
        plt.ylabel("频率")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/error_distribution.png")
        plt.close()

        # 预瞄距离与曲率关系图
        plt.figure(figsize=(10, 6))
        # 确保两个数组长度一致
        min_length = min(len(self.current_curvatures), len(self.lookahead_distances))
        if min_length > 0:
            plt.scatter(self.current_curvatures[:min_length], self.lookahead_distances[:min_length], s=5, c='purple',
                        alpha=0.5)
            plt.title("预瞄距离与路径曲率关系")
            plt.xlabel("路径曲率")
            plt.ylabel("预瞄距离 (m)")
            plt.grid(True)
            plt.savefig(f"{self.output_dir}/lookahead_vs_curvature.png")
        plt.close()

        # 保存局部放大图的快照
        plt.figure(figsize=(8, 6))
        plt.plot([wp[0] for wp in self.waypoints], [wp[1] for wp in self.waypoints], 'b-', alpha=0.5)

        # 绘制轨迹
        if self.vehicle.trajectory:
            traj_x, traj_y, _ = zip(*self.vehicle.trajectory)
            plt.plot(traj_x, traj_y, 'g-', alpha=0.6)

        # 绘制车辆最后位置
        if self.vehicle.trajectory:
            last_x, last_y, _ = self.vehicle.trajectory[-1]
            plt.plot(last_x, last_y, 'ro', markersize=8)

        # 设置局部放大区域
        if self.vehicle.trajectory:
            last_x, last_y, _ = self.vehicle.trajectory[-1]
            plt.xlim(last_x - 5, last_x + 5)
            plt.ylim(last_y - 5, last_y + 5)

        plt.title("仿真结束时的局部视图")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(f"{self.output_dir}/final_zoom_view.png")
        plt.close()

        print(f"性能图表已保存到 {self.output_dir} 目录")

    def run(self):
        """运行仿真"""
        # 初始位置设置 (靠近第一个路径点)
        if self.waypoints:
            start_x, start_y = self.waypoints[0]
            self.vehicle.state[0] = start_x
            self.vehicle.state[1] = start_y  # 从路径下方开始

            # 设置初始朝向指向路径方向
            if len(self.waypoints) > 1:
                dx = self.waypoints[1][0] - start_x
                dy = self.waypoints[1][1] - start_y
                initial_yaw = math.atan2(dy, dx)
                self.vehicle.state[2] = initial_yaw

            # 设置初始速度
            self.vehicle.state[3] = self.controller.target_speed

            # 创建KDTree用于查找最近点
            self.waypoint_tree = KDTree(self.waypoints)

        # 根据SHOW_ANIMATION选项决定是否创建和显示动画
        if SHOW_ANIMATION:
            # 创建动画
            self.ani = animation.FuncAnimation(
                self.fig, self.update, frames=self.max_steps,
                init_func=self.init_animation,
                interval=20, blit=True)

            plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)  # 增加内边距
            plt.show()

            # 新增：如果到达终点但动画仍在运行，手动停止
            if self.reached_end and self.ani and self.ani.event_source:
                self.ani.event_source.stop()
        else:
            # 不显示动画，直接运行仿真
            print("正在运行仿真（无动画显示）...")
            for frame in range(self.max_steps):
                # 更新仿真状态
                self.update(frame)

                # 检查是否到达终点
                if self.reached_end:
                    break

            # 仿真结束后创建性能图表（只有在未调用finalize_simulation时才调用）
            # 注意：如果在update中已经调用了finalize_simulation（例如到达终点时），则不再重复调用
            print("仿真完成，结果已保存。")


if __name__ == "__main__":
    sim = Simulation()
    sim.run()