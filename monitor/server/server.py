import socket
import threading
import time
import json
import numpy as np
from collections import deque
from typing import Dict, Any, List
import cantools
import can

# ========================
# 配置参数
# ========================
DBC_FILE = '20250626.dbc'
CAN_CHANNEL = 0
BITRATE = 500000
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 12345
TRAJECTORY_QUEUE_SIZE = 100  # 积分轨迹队列最大长度
TARGET_SIGNALS = [
    'SGW_IVI_GyroX', 'SGW_IVI_GyroY', 'SGW_IVI_GyroZ',
    'SGW_IVI_AccelX', 'SGW_IVI_AcceY', 'SGW_IVI_AccelZ',
    'ACU_YawRateSt', 'IBC_VehicleSpeed', 'TAS_SAS_SteeringAngle'
]


class SocketServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients: List[socket.socket] = []
        self.client_lock = threading.Lock()
        self.running = False

    def start(self):
        """启动服务器"""
        if self.running:
            return
        self.running = True
        thread = threading.Thread(target=self._server_loop, daemon=True)
        thread.start()
        print(f"SocketServer: 启动于 {self.host}:{self.port}")

    def stop(self):
        """停止服务器"""
        self.running = False
        with self.client_lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        if self.server_socket:
            self.server_socket.close()

    def _server_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)
            print(f"SocketServer: 监听 {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    print(f"新客户端连接: {addr}")

                    with self.client_lock:
                        self.clients.append(client_socket)

                    # 为客户端启动接收线程（可选：用于接收心跳或命令）
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr),
                        daemon=True
                    )
                    client_thread.start()

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"SocketServer 错误: {e}")
                    break

        except Exception as e:
            print(f"无法绑定服务器: {e}")
        finally:
            self.stop()

    def _handle_client(self, client_socket: socket.socket, addr):
        """处理单个客户端（保持连接）"""
        try:
            while self.running:
                # 可以接收客户端消息（如心跳、请求）
                client_socket.settimeout(1.0)
                try:
                    data = client_socket.recv(1024)
                    if not data:
                        print("_handle_client not data break")
                        break
                except socket.timeout:
                    continue
        except Exception as e:
            print(f"客户端 {addr} 通信错误: {e}")
        finally:
            with self.client_lock:
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
            try:
                client_socket.close()
            except:
                pass
            print(f"客户端断开: {addr}")

    def broadcast(self, message: Dict[str, Any]):
        """广播消息给所有客户端"""
        if not self.running:
            return

        json_data = json.dumps(message) + '\n'  # 添加换行符作为分隔
        dead_clients = []

        with self.client_lock:
            for client in self.clients:
                try:
                    client.sendall(json_data.encode('utf-8'))
                except Exception as e:
                    print(f"广播失败 (客户端断开?): {e}")
                    dead_clients.append(client)

            # 清理断开的客户端
            for client in dead_clients:
                if client in self.clients:
                    self.clients.remove(client)


