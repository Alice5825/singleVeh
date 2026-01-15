import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle, Polygon
import math

# 全局常量
DT = 0.05  # 控制周期
WHEELBASE = 0.144  # 轴距
SIM_TIME = 20.0  # 模拟时间


class PIDController:
    """通用的PID控制器"""

    def __init__(self, kp, ki, kd, max_output=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output

        self.integral = 0.0
        self.prev_error = 0.0
        self.output = 0.0

    def update(self, error, dt):
        """更新PID控制器"""
        # 积分项
        self.integral += error * dt

        # 微分项
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        # PID输出
        self.output = (
                self.kp * error +
                self.ki * self.integral +
                self.kd * derivative
        )

        # 限制输出
        self.output = np.clip(self.output, -self.max_output, self.max_output)

        # 保存当前误差
        self.prev_error = error

        return self.output


class AckermannVehicle:
    """阿克曼小车模型"""

    def __init__(self, wheelbase=WHEELBASE,
                 v_pid=None, w_pid=None,
                 max_steer=0.7, max_vel=2.0):
        # 初始状态 [x, y, yaw, v, steer]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.wheelbase = wheelbase

        # PID控制器
        self.v_pid = v_pid or PIDController(kp=2.0, ki=0.1, kd=0.05)
        self.w_pid = w_pid or PIDController(kp=3.0, ki=0.05, kd=0.02)

        # 限制
        self.max_steer = max_steer
        self.max_vel = max_vel

        # 历史记录
        self.trajectory = []
        self.time_history = []
        self.v_history = []
        self.w_history = []
        self.target_v_history = []
        self.target_w_history = []

    def update(self, target_v, target_w, dt=DT, control_mode='v'):
        """更新车辆状态

        Parameters:
        -----------
        target_v : float
            目标速度 (m/s)
        target_w : float
            目标角速度 (rad/s)
        dt : float
            时间步长
        control_mode : str
            'v': 控制速度v
            'w': 控制角速度w
        """
        x, y, yaw, v, steer = self.state

        # 根据目标角速度计算目标转向角
        if abs(v) > 0.01:  # 避免除零
            target_steer = np.arctan(target_w * self.wheelbase / v)
            target_steer = np.clip(target_steer, -self.max_steer, self.max_steer)
        else:
            target_steer = steer

        if control_mode == 'v':
            # 控制速度v
            v_error = target_v - v
            v_control = self.v_pid.update(v_error, dt)

            # 转向控制（使用简单的P控制）
            steer_error = target_steer - steer
            steer_control = 3.0 * steer_error  # 简单P控制
            steer_control = np.clip(steer_control, -3.0, 3.0)

        else:  # control_mode == 'w'
            # 控制角速度w
            current_w = v * np.tan(steer) / self.wheelbase
            w_error = target_w - current_w

            # 将角速度误差转换为转向角误差
            if abs(v) > 0.01:
                steer_error = np.arctan(w_error * self.wheelbase / v)
            else:
                steer_error = 0.0

            steer_control = self.w_pid.update(steer_error, dt)

            # 速度控制（使用简单的P控制）
            v_error = target_v - v
            v_control = 1.0 * v_error  # 简单P控制

        # 更新速度和转向角
        v += v_control * dt
        v = np.clip(v, -self.max_vel, self.max_vel)

        steer += steer_control * dt
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        # 计算实际角速度
        if abs(v) > 0.01:
            actual_w = v * np.tan(steer) / self.wheelbase
        else:
            actual_w = 0.0

        # 更新位置和朝向
        beta = np.arctan(0.5 * np.tan(steer))  # 阿克曼转向近似
        yaw += actual_w * dt
        x += v * np.cos(yaw + beta) * dt
        y += v * np.sin(yaw + beta) * dt

        # 归一化角度
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

        # 更新状态
        self.state = np.array([x, y, yaw, v, steer])

        # 记录历史
        self.trajectory.append((x, y, yaw))
        self.v_history.append(v)
        self.w_history.append(actual_w)
        self.target_v_history.append(target_v)
        self.target_w_history.append(target_w)

        return actual_w


def create_vehicle_visualization_control(control_mode='v'):
    """创建阿克曼小车可视化控制系统

    Parameters:
    -----------
    control_mode : str
        'v': 控制速度v
        'w': 控制角速度w
    """
    # 设置matplotlib后端，避免警告
    plt.rcParams['backend'] = 'TkAgg'

    # 创建车辆实例
    if control_mode == 'v':
        v_pid = PIDController(kp=1.5, ki=0.3, kd=0.1)
        w_pid = PIDController(kp=1.0, ki=0.05, kd=0.02)
    else:  # 'w'
        v_pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        w_pid = PIDController(kp=2.0, ki=0.1, kd=0.1)

    vehicle = AckermannVehicle(
        wheelbase=WHEELBASE,
        v_pid=v_pid,
        w_pid=w_pid
    )

    # 创建图形和子图
    fig = plt.figure(figsize=(15, 8))

    # 1. 轨迹图
    ax_traj = plt.subplot(2, 3, (1, 4))
    ax_traj.set_xlabel('X (m)')
    ax_traj.set_ylabel('Y (m)')
    ax_traj.set_title('Vehicle Trajectory')
    ax_traj.grid(True)
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.set_xlim(-5, 5)
    ax_traj.set_ylim(-5, 5)

    # 2. 速度/角速度曲线
    ax_plot = plt.subplot(2, 3, (2, 5))
    if control_mode == 'v':
        ax_plot.set_ylabel('Velocity (m/s)')
        ax_plot.set_title('Velocity Control')
    else:
        ax_plot.set_ylabel('Angular Velocity (rad/s)')
        ax_plot.set_title('Angular Velocity Control')
    ax_plot.set_xlabel('Time (s)')
    ax_plot.grid(True)

    # 3. PID参数调整滑块区域
    ax_sliders = plt.subplot(2, 3, (3, 6))
    ax_sliders.axis('off')  # 隐藏坐标轴

    # 初始化图形元素
    traj_line, = ax_traj.plot([], [], 'b-', lw=2, alpha=0.6, label='Trajectory')

    # 用三角形表示车辆，包含方向信息
    vehicle_triangle = Polygon([[0, 0], [0, 0], [0, 0]],
                               closed=True, fill=True,
                               color='red', alpha=0.8)
    ax_traj.add_patch(vehicle_triangle)

    if control_mode == 'v':
        actual_line, = ax_plot.plot([], [], 'b-', lw=2, label='Actual Velocity')
        target_line, = ax_plot.plot([], [], 'r--', lw=2, label='Target Velocity')
    else:
        actual_line, = ax_plot.plot([], [], 'b-', lw=2, label='Actual Angular Velocity')
        target_line, = ax_plot.plot([], [], 'r--', lw=2, label='Target Angular Velocity')

    ax_plot.legend(loc='upper right')

    # 创建滑块（根据控制模式显示不同的PID参数）
    slider_height = 0.03
    slider_spacing = 0.05

    if control_mode == 'v':
        # 速度PID参数
        ax_kp = plt.axes([0.72, 0.7, 0.25, slider_height])
        ax_ki = plt.axes([0.72, 0.65, 0.25, slider_height])
        ax_kd = plt.axes([0.72, 0.6, 0.25, slider_height])
        # here
        slider_kp = Slider(ax_kp, 'Kp_v', 0.0, 10.0, valinit=v_pid.kp)
        slider_ki = Slider(ax_ki, 'Ki_v', 0.0, 2.0, valinit=v_pid.ki)
        slider_kd = Slider(ax_kd, 'Kd_v', 0.0, 2.0, valinit=v_pid.kd)

        # 目标速度控制
        ax_target_v = plt.axes([0.72, 0.55, 0.25, slider_height])
        slider_target_v = Slider(ax_target_v, 'Target V', -1.0, 1.0, valinit=0.5)

        # 目标角速度（用于转向）
        ax_target_w = plt.axes([0.72, 0.5, 0.25, slider_height])
        slider_target_w = Slider(ax_target_w, 'Target W', -2.0, 2.0, valinit=0.5)

    else:  # control_mode == 'w'
        # 角速度PID参数
        ax_kp = plt.axes([0.72, 0.7, 0.25, slider_height])
        ax_ki = plt.axes([0.72, 0.65, 0.25, slider_height])
        ax_kd = plt.axes([0.72, 0.6, 0.25, slider_height])

        slider_kp = Slider(ax_kp, 'Kp_w', 0.0, 10.0, valinit=w_pid.kp)
        slider_ki = Slider(ax_ki, 'Ki_w', 0.0, 2.0, valinit=w_pid.ki)
        slider_kd = Slider(ax_kd, 'Kd_w', 0.0, 2.0, valinit=w_pid.kd)

        # 目标角速度
        ax_target_w = plt.axes([0.72, 0.55, 0.25, slider_height])
        slider_target_w = Slider(ax_target_w, 'Target W', -2.0, 2.0, valinit=0.5)

        # 目标速度
        ax_target_v = plt.axes([0.72, 0.5, 0.25, slider_height])
        slider_target_v = Slider(ax_target_v, 'Target V', -1.0, 1.0, valinit=0.5)

    # 重置按钮
    reset_ax = plt.axes([0.72, 0.4, 0.25, 0.04])
    reset_button = Button(reset_ax, 'Reset Simulation', color='lightgoldenrodyellow')

    # 切换模式按钮
    if control_mode == 'v':
        switch_ax = plt.axes([0.72, 0.35, 0.25, 0.04])
        switch_button = Button(switch_ax, 'Switch to W Control', color='lightblue')
    else:
        switch_ax = plt.axes([0.72, 0.35, 0.25, 0.04])
        switch_button = Button(switch_ax, 'Switch to V Control', color='lightblue')

    # 全局变量
    time_elapsed = 0
    simulation_running = True
    current_arrow = None  # 跟踪当前箭头对象

    def update_pid_parameters():
        """更新PID参数"""
        if control_mode == 'v':
            vehicle.v_pid.kp = slider_kp.val
            vehicle.v_pid.ki = slider_ki.val
            vehicle.v_pid.kd = slider_kd.val
        else:
            vehicle.w_pid.kp = slider_kp.val
            vehicle.w_pid.ki = slider_ki.val
            vehicle.w_pid.kd = slider_kd.val

    def reset_simulation(event=None):
        """重置仿真"""
        nonlocal time_elapsed

        # 重置车辆状态
        vehicle.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        vehicle.trajectory = []
        vehicle.time_history = []
        vehicle.v_history = []
        vehicle.w_history = []
        vehicle.target_v_history = []
        vehicle.target_w_history = []

        # 重置PID控制器
        if control_mode == 'v':
            vehicle.v_pid.integral = 0.0
            vehicle.v_pid.prev_error = 0.0
        else:
            vehicle.w_pid.integral = 0.0
            vehicle.w_pid.prev_error = 0.0

        time_elapsed = 0

        # 更新图形
        traj_line.set_data([], [])
        actual_line.set_data([], [])
        target_line.set_data([], [])

        # 重置车辆三角形位置
        vehicle_triangle.set_xy([[0, 0], [0, 0], [0, 0]])

        # 清除之前的箭头
        nonlocal current_arrow
        if current_arrow is not None:
            try:
                current_arrow.remove()
            except:
                pass
            current_arrow = None

    def switch_control_mode(event=None):
        """切换控制模式"""
        nonlocal simulation_running
        simulation_running = False
        plt.close(fig)

        # 稍微延迟一下，确保窗口关闭
        import time
        time.sleep(0.1)

        new_mode = 'w' if control_mode == 'v' else 'v'
        create_vehicle_visualization_control(new_mode)

    def update_vehicle_shape():
        """更新车辆图形"""
        x, y, yaw, v, steer = vehicle.state

        # 车辆尺寸
        length = 0.21
        width = 0.18

        # 计算车辆三角形顶点（等边三角形，一个顶点朝前）
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # 三角形的三个顶点：前点、左后点、右后点
        front_point = [x + length / 2 * cos_yaw, y + length / 2 * sin_yaw]
        left_back = [x - length / 2 * cos_yaw + width / 2 * sin_yaw,
                     y - length / 2 * sin_yaw - width / 2 * cos_yaw]
        right_back = [x - length / 2 * cos_yaw - width / 2 * sin_yaw,
                      y - length / 2 * sin_yaw + width / 2 * cos_yaw]

        # 更新车辆三角形
        vehicle_triangle.set_xy([front_point, left_back, right_back])

        # 添加方向箭头
        arrow_length = 0.15
        arrow_x = x + 0.1 * cos_yaw
        arrow_y = y + 0.1 * sin_yaw

        # 清除之前的箭头
        nonlocal current_arrow
        if current_arrow is not None:
            try:
                current_arrow.remove()
            except:
                pass

        # 创建新的箭头
        current_arrow = ax_traj.arrow(
            arrow_x, arrow_y,
            arrow_length * cos_yaw, arrow_length * sin_yaw,
            head_width=0.05, head_length=0.1,
            fc='green', ec='green', zorder=5
        )

    def update_plot(frame):
        """更新图形 - 动画回调函数"""
        nonlocal time_elapsed

        # 获取目标值
        target_v = slider_target_v.val
        target_w = slider_target_w.val

        if simulation_running and time_elapsed < SIM_TIME:
            # 更新PID参数
            update_pid_parameters()

            # 更新车辆状态
            vehicle.update(target_v, target_w, DT, control_mode)
            time_elapsed += DT

            # 记录时间
            vehicle.time_history.append(time_elapsed)

        # 更新轨迹图
        if vehicle.trajectory:
            traj_data = np.array(vehicle.trajectory)
            traj_line.set_data(traj_data[:, 0], traj_data[:, 1])

            # 动态调整轨迹图范围
            if len(traj_data) > 1:
                margin = 2.0
                x_min, x_max = traj_data[:, 0].min(), traj_data[:, 0].max()
                y_min, y_max = traj_data[:, 1].min(), traj_data[:, 1].max()

                ax_traj.set_xlim(min(-5, x_min - margin), max(5, x_max + margin))
                ax_traj.set_ylim(min(-5, y_min - margin), max(5, y_max + margin))

        # 更新车辆形状
        update_vehicle_shape()

        # 更新速度/角速度曲线
        if vehicle.time_history:
            times = vehicle.time_history
            if control_mode == 'v':
                actual_line.set_data(times, vehicle.v_history)
                target_line.set_data(times, vehicle.target_v_history)
                ax_plot.set_ylim(-1.2, 1.2)
            else:
                actual_line.set_data(times, vehicle.w_history)
                target_line.set_data(times, vehicle.target_w_history)
                ax_plot.set_ylim(-2.5, 2.5)

            ax_plot.set_xlim(0, max(times[-1] + 0.1, SIM_TIME))

        return traj_line, vehicle_triangle, actual_line, target_line

    def on_close(event):
        """关闭窗口时的回调函数"""
        nonlocal simulation_running
        simulation_running = False

    # 连接事件
    fig.canvas.mpl_connect('close_event', on_close)
    reset_button.on_clicked(reset_simulation)
    switch_button.on_clicked(switch_control_mode)

    # 初始重置
    reset_simulation()

    # 创建动画
    ani = animation.FuncAnimation(
        fig,
        update_plot,
        interval=50,  # 50ms更新一次
        blit=False,
        cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()

    return vehicle, fig, ani


# 使用示例
if __name__ == "__main__":
    # 可以选择 'v' 或 'w' 控制模式
    print("Starting vehicle simulation...")
    print("Mode 'v': Velocity PID control")
    print("Mode 'w': Angular velocity PID control")

    # 默认以速度控制模式启动
    try:
        vehicle, fig, ani = create_vehicle_visualization_control(control_mode='v')
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()