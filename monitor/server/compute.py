import socket
import threading
import time
import json
import numpy as np
from collections import deque
from typing import Dict, Any, List
import cantools
import can
from monitor.server.can_reader import CANReader
import math 



class TrajectoryIntegrator:
    def __init__(self,can_reader:CANReader,  queue_size: int = 100):
        self.can_reader = can_reader
        self.queue_size = queue_size
        self.trajectory_queue = deque(maxlen=queue_size)  # 自动FIFO

        # 状态变量
        self.current_position = [0.0, 0.0]  # x, y
        self.current_velocity_x = 0.0
        self.current_velocity_y = 0.0
        self.current_heading = 0.0  # 弧度
        self.current_speed = 0.0
        self.last_timestamp = time.time()
        self.running = False
        self._lock = threading.Lock()

        self.debug_count = 0

    def start(self):
        """启动积分处理线程"""
        if self.running:
            return
        self.running = True
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
        print("TrajectoryIntegrator: 启动积分计算线程")

    def stop(self):
        """停止积分线程"""
        self.running = False

    def put_data(self, data: Dict[str, Any]):
        """从CAN线程接收原始数据"""
        if not self.running:
            return
        self._integrate(data)

    def _integrate(self, data: Dict[str, Any]):
        if data is None:
            return
        """核心积分逻辑"""
        timestamp = data['timestamp']
        signals = data['signals']

        current_time = timestamp
        dt = current_time - self.last_timestamp
        if dt <= 0:
            current_time = self.last_timestamp + 0.1
            timestamp = current_time
            dt = 0.1
        elif dt > 1.0:
            print(f"警告: 时间间隔过大 dt={dt:.3f}s，可能是数据跳跃")
            dt = 0.1
        self.last_timestamp = current_time

        self.debug_count += 1



        # 单位转换
        vehicle_speed = signals.get('IBC_VehicleSpeed', 0) / 3.6  # km/h -> m/s
        sgw_accel_x = signals.get('SGW_IVI_AccelX', 0) * 2e-5
        sgw_accel_y = signals.get('SGW_IVI_AcceY', 0) * 2e-5
        sgw_accel_z = signals.get('SGW_IVI_AccelZ', 0) * 2e-5
        sgw_gyro_x = signals.get('SGW_IVI_GyroX', 0) * 0.001
        sgw_gyro_y = signals.get('SGW_IVI_GyroY', 0) * 0.001
        sgw_gyro_z = signals.get('SGW_IVI_GyroZ', 0) * 0.001
        acu_yaw_rate = signals.get('ACU_YawRateSt', 0) * np.pi / 180  # 示例转换
        steering_angle = signals.get('TAS_SAS_SteeringAngle', 0) / 16.0
        pdcu_actual_gear = signals.get('PDCU_ActualGear',4) # default D , P:5 ,R:7, N:0,D:4

        if pdcu_actual_gear == 7:
            vehicle_speed = -vehicle_speed
            sgw_accel_x=-sgw_accel_x
            sgw_accel_y=-sgw_accel_y
        # 更新航向角
        if acu_yaw_rate != 0:
            self.current_heading += acu_yaw_rate * dt
        else:
            self.current_heading += sgw_gyro_z * dt

        # 计算速度
        if abs(vehicle_speed) > 0.1:  # 使用车速
            velocity_x = vehicle_speed * np.cos(self.current_heading)
            velocity_y = vehicle_speed * np.sin(self.current_heading)
        else:  # 积分加速度
            velocity_x = self.current_velocity_x + sgw_accel_x * dt
            velocity_y = self.current_velocity_y + sgw_accel_y * dt

        self.current_velocity_x = velocity_x
        self.current_velocity_y = velocity_y

        # 更新位置
        dx = velocity_x * dt
        dy = velocity_y * dt
        self.current_position[0] += dx
        self.current_position[1] += dy
        self.current_speed = vehicle_speed

        # 构造轨迹数据
        trajectory_data = {
            "timestamp": timestamp,
            "position": {"x": self.current_position[0], "y": self.current_position[1]},
            "velocity": {"x": velocity_x, "y": velocity_y},
            "heading": self.current_heading,
            "speed": self.current_speed,
            "raw_signals": signals
        }

        # 存入队列（自动FIFO）
        with self._lock:
            self.trajectory_queue.append(trajectory_data)

        # 调试输出（每20条）
        if self.debug_count % 20 == 0:
            total_displacement = np.sqrt(self.current_position[0]**2 + self.current_position[1]**2)
            print(f"[积分调试] 速度: {vehicle_speed:.3f} m/s, dt: {dt:.6f}s")
            print(f"  位置: ({self.current_position[0]:.3f}, {self.current_position[1]:.3f}), 总位移: {total_displacement:.3f}m")
            print(f"  航向角: {np.degrees(self.current_heading):.2f}°")

    def _process_loop(self):
        while self.running:
            self.put_data(self.can_reader.get_latest_data())
            time.sleep(0.1)

    @property
    def lock(self):
        return self._lock




