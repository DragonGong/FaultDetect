import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import can
import cantools
from collections import defaultdict
import socket
import json
import threading
import time
import queue
from .client import DataClient


class Visualizer:

    def __init__(self, host='127.0.0.1', port=12345):

        self.debug_count = 0
        self.host = host
        self.port = port

        # 数据队列
        self.data_queue = queue.Queue(maxsize=10)
        self.running = False

        # 车辆状态
        self.current_position = np.array([0.0, 0.0])
        self.current_heading = 0.0
        self.current_speed = 0.0
        self.current_timestamp = 0.0
        self.current_velocity_x = 0.0
        self.current_velocity_y = 0.0

        # 轨迹数据
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_heading = []
        self.trajectory_speed = []
        self.timestamps = []

        # 车辆参数
        self.vehicle_length = 4.0
        self.vehicle_width = 1.8

        # 轨迹历史长度
        self.trajectory_history_length = 100

        # 关键信号名称
        self.key_signals = [
            'SGW_IVI_GyroX',  # X轴角速度
            'SGW_IVI_GyroY',  # Y轴角速度
            'SGW_IVI_GyroZ',  # Z轴角速度 (主要横摆角速度)
            'SGW_IVI_AccelX',  # X轴加速度
            'SGW_IVI_AcceY',  # Y轴加速度
            'SGW_IVI_AccelZ',  # Z轴加速度
            'ACU_YawRateSt',  # 横摆角速度
            'IBC_VehicleSpeed',  # 车辆速度
            'TAS_SAS_SteeringAngle'  # 转向角
        ]

        # 初始化信号值
        self.signal_values = {signal: 0.0 for signal in self.key_signals}
        self.last_timestamp = time.time()


        self.client = DataClient(host=self.host, port=self.port, queue_size=100)

    def process_realtime_data(self):
        # 直接更新数据
        data = self.client.get_latest_data()
        if data is None:
            print("data is none")
            return
        else:
            print(f"data is {data}")
        self.current_heading = data["heading"]
        self.current_position = np.array([data["position"]["x"], data["position"]["y"]])
        self.current_timestamp = data["timestamp"]
        self.current_speed = data["speed"]

        # 调试位置计算
        if self.debug_count % 20 == 0:  # 每20个数据点打印一次
            total_displacement = np.sqrt(self.current_position[0] ** 2 + self.current_position[1] ** 2)
            print(f"  当前位置: ({self.current_position[0]:.3f}, {self.current_position[1]:.3f})")
            print(f"  总位移: {total_displacement:.3f} m")
            print("---")
        self.debug_count += 1
        # 添加到轨迹历史
        self.trajectory_x.append(self.current_position[0])
        self.trajectory_y.append(self.current_position[1])
        self.trajectory_heading.append(self.current_heading)
        self.trajectory_speed.append(self.current_speed)
        self.timestamps.append(self.current_timestamp)

        # 限制轨迹历史长度
        if len(self.trajectory_x) > self.trajectory_history_length:
            self.trajectory_x.pop(0)
            self.trajectory_y.pop(0)
            self.trajectory_heading.pop(0)
            self.trajectory_speed.pop(0)
            self.timestamps.pop(0)


    def setup_realtime_plot(self):
        """设置实时绘图环境"""
        self.fig, (self.ax_main, self.ax_speed) = plt.subplots(1, 2, figsize=(16, 8))

        # 主轨迹图
        self.ax_main.set_xlabel('X Position (m)', fontsize=12)
        self.ax_main.set_ylabel('Y Position (m)', fontsize=12)
        self.ax_main.set_title('Real-time Vehicle Trajectory (Socket Data)', fontsize=14, fontweight='bold')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')

        # 速度图
        self.ax_speed.set_xlabel('Time (s)', fontsize=12)
        self.ax_speed.set_ylabel('Speed (m/s)', fontsize=12)
        self.ax_speed.set_title('Vehicle Speed vs Time', fontsize=14, fontweight='bold')
        self.ax_speed.grid(True, alpha=0.3)

        # 初始化轨迹线
        self.trajectory_line, = self.ax_main.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Trajectory')

        # 初始化车辆图标
        self.vehicle_patch = self.create_vehicle_shape()
        self.ax_main.add_patch(self.vehicle_patch)

        # 初始化方向箭头
        self.heading_arrow = self.ax_main.arrow(0, 0, 0, 0, head_width=0.5, head_length=0.3,
                                                fc='red', ec='red', alpha=0.8)

        # 初始化速度线
        self.speed_line, = self.ax_speed.plot([], [], 'g-', linewidth=2, label='Speed')

        # 添加信息文本
        self.info_text = self.ax_main.text(0.02, 0.98, '', transform=self.ax_main.transAxes,
                                           fontsize=10, verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 添加图例
        self.ax_main.legend(loc='upper right')
        self.ax_speed.legend(loc='upper right')

        # 调整布局
        plt.tight_layout()

    def create_vehicle_shape(self, x=0, y=0, heading=0):
        """创建车辆形状"""
        from matplotlib.patches import Polygon

        # 车辆尺寸
        length = self.vehicle_length
        width = self.vehicle_width

        # 创建车辆主体 (梯形形状)
        front_width = width * 0.7
        rear_width = width

        # 车辆顶点坐标 (相对于中心点)
        vehicle_vertices = np.array([
            [length / 2, front_width / 2],  # 前左
            [0, rear_width / 2],  # 中左
            [-length / 2, rear_width / 2],  # 后左
            [-length / 2, -rear_width / 2],  # 后右
            [0, -rear_width / 2],  # 中右
            [length / 2, -front_width / 2],  # 前右
        ])

        # 旋转车辆顶点
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        rotated_vertices = vehicle_vertices @ rotation_matrix.T

        # 平移到指定位置
        vehicle_vertices = rotated_vertices + np.array([x, y])

        # 创建车辆主体
        vehicle_body = Polygon(vehicle_vertices,
                               facecolor='lightblue',
                               edgecolor='blue',
                               linewidth=2,
                               alpha=0.8)

        return vehicle_body

    def update_realtime_frame(self, frame):
        """更新实时动画帧"""
        # 处理新数据
        self.process_realtime_data()

        if len(self.trajectory_x) == 0:
            return (self.trajectory_line, self.vehicle_patch, self.heading_arrow,
                    self.speed_line, self.info_text)

        # 更新当前位置
        x = self.current_position[0]
        y = self.current_position[1]
        heading = self.current_heading
        speed = self.current_speed
        timestamp = self.current_timestamp

        # 更新轨迹线
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)

        # 更新车辆位置和方向
        self.vehicle_patch.remove()
        self.vehicle_patch = self.create_vehicle_shape(x, y, heading)
        self.ax_main.add_patch(self.vehicle_patch)

        # 更新方向箭头
        arrow_length = 1.6
        dx = arrow_length * np.cos(heading)
        dy = arrow_length * np.sin(heading)

        self.heading_arrow.remove()
        self.heading_arrow = self.ax_main.arrow(x, y, dx, dy, head_width=0.5, head_length=0.3,
                                                fc='red', ec='red', alpha=0.8)

        # 更新速度图
        if len(self.timestamps) > 1:
            self.speed_line.set_data(self.timestamps, self.trajectory_speed)

        # 更新信息文本
        info_text = f'Time: {timestamp:.2f}s\n'
        info_text += f'Position: ({x:.2f}, {y:.2f}) m\n'
        info_text += f'Speed: {speed:.2f} m/s ({speed * 3.6:.1f} km/h)\n'
        info_text += f'Heading: {np.degrees(heading):.1f}°\n'
        info_text += f'Data Points: {len(self.trajectory_x)}\n'
        info_text += f'Data Source: Socket'

        self.info_text.set_text(info_text)

        # 动态调整主图视图范围
        if len(self.trajectory_x) > 1:
            margin = 5
            self.ax_main.set_xlim(min(self.trajectory_x) - margin, max(self.trajectory_x) + margin)
            self.ax_main.set_ylim(min(self.trajectory_y) - margin, max(self.trajectory_y) + margin)

        # 动态调整速度图范围
        if len(self.timestamps) > 1:
            self.ax_speed.set_xlim(min(self.timestamps), max(self.timestamps))
            self.ax_speed.set_ylim(min(self.trajectory_speed) - 0.5, max(self.trajectory_speed) + 0.5)

        return (self.trajectory_line, self.vehicle_patch, self.heading_arrow,
                self.speed_line, self.info_text)



    def start_realtime_animation(self, interval=50):
        """开始实时动画"""
        print("开始实时轨迹动画...")

        # 启动数据接收
        self.start_data_receiver()

        # 创建动画
        anim = animation.FuncAnimation(
            self.fig,
            self.update_realtime_frame,
            interval=interval,
            blit=False,
            repeat=True
        )

        plt.show()
        return anim

    def start_data_receiver(self):
        self.client.start()

    def run_realtime_visualization(self, interval=50):
        """运行实时可视化"""
        print("=== 实时车辆轨迹可视化系统 (Socket数据) ===")

        # 设置绘图
        self.setup_realtime_plot()

        # 开始动画
        anim = self.start_realtime_animation(interval=interval)

        return anim