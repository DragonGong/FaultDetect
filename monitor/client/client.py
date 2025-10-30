import socket
import json
import time
import threading
import queue
from collections import deque
from typing import Dict, Any, Optional

class DataClient:
    def __init__(self, host: str, port: int, queue_size: int = 10, key_signals=None):
        """
        初始化客户端
        :param host: 服务器地址
        :param port: 服务器端口
        :param queue_size: 本地缓存的最大数据条数（FIFO）
        """
        if key_signals is None:
            key_signals = [
                "timestamp",
                "position",  # 其值是包含"x"和"y"的字典
                "velocity",  # 其值是包含"x"和"y"的字典
                "heading",
                "speed",
                "raw_signals"
            ]
        self.host = host
        self.port = port
        self.running = False
        self.data_queue = deque(maxlen=queue_size)  # 使用 deque 自动 FIFO
        self.key_signals = key_signals  # 可根据需要设置关注的信号名
        self.signal_values = {}  # 存储当前信号最新值
        self._lock = threading.Lock()  # 保护 data_queue 的线程锁（虽然 deque 是线程安全的，但操作复合时建议加锁）
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """启动客户端，开启后台接收线程"""
        if self.running:
            print("客户端已在运行中...")
            return

        self.running = True
        self._thread = threading.Thread(target=self._receive_data, daemon=True)
        self._thread.start()
        print(f"客户端已启动，连接至 {self.host}:{self.port}")

    def stop(self):
        """停止客户端"""
        if not self.running:
            return
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("客户端已停止")

    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        获取队列中最新的数据（不删除）
        :return: 最新的一条数据 dict，或 None（如果为空）
        """
        with self._lock:
            if len(self.data_queue) > 0:
                return self.data_queue[-1]  # 返回最后一个元素（最新）
            return None

    def _receive_data(self):
        """后台线程运行：接收数据并处理"""
        client_socket = None
        data_buffer = ""  # 缓冲区：存储未解析的剩余数据

        while self.running:
            try:
                # 建立连接
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(5.0)
                client_socket.connect((self.host, self.port))
                client_socket.settimeout(0.1)  # 接收超时
                print(f"已连接到服务器 {self.host}:{self.port}")

                while self.running:
                    try:
                        # 检查接收缓冲区大小，防止积压
                        buffer_size = client_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                        if buffer_size > 8192:
                            # 丢弃旧数据，只保留最后 4KB
                            try:
                                client_socket.recv(buffer_size - 4096, socket.MSG_PEEK)
                                client_socket.recv(buffer_size - 4096)
                            except:
                                pass  # 忽略丢弃过程中的异常
                            data_buffer = ""  # 清空解析缓冲

                        # 接收新数据
                        recv_data = client_socket.recv(4096)
                        if not recv_data:
                            print("服务器主动断开连接，尝试重连...")
                            break

                        recv_str = recv_data.decode('utf-8', errors='ignore')
                        data_buffer += recv_str

                        # 按行分割，提取完整 JSON
                        json_lines = []
                        while '\n' in data_buffer:
                            json_line, data_buffer = data_buffer.split('\n', 1)
                            json_line = json_line.strip()
                            if json_line:
                                json_lines.append(json_line)

                        # 只保留最多最后 5 条（可配置）
                        latest_lines = json_lines[-5:] if len(json_lines) > 5 else json_lines

                        for json_line in latest_lines:
                            try:
                                message_data = json.loads(json_line)

                                signal_values= {}
                                # 更新当前信号值
                                for signal_name in self.key_signals:
                                    if signal_name in message_data:
                                        signal_values[signal_name] = message_data[signal_name]
                                # 使用 deque，自动 FIFO
                                with self._lock:
                                    # print(f"client receive data :{signal_values}")
                                    self.data_queue.append(signal_values)

                            except json.JSONDecodeError as e:
                                print(f"JSON解析失败（行：{json_line[:50]}...）: {e}")
                            except Exception as e:
                                print(f"数据处理失败: {e}")

                    except socket.timeout:
                        continue
                    except ConnectionResetError:
                        print("服务器强制断开连接，尝试重连...")
                        break
                    except Exception as e:
                        print(f"接收循环错误: {e}")
                        break

            except ConnectionRefusedError:
                print(f"无法连接到服务器 {self.host}:{self.port}，2秒后重试...")
            except socket.timeout:
                print(f"连接服务器超时，2秒后重试...")
            except Exception as e:
                print(f"连接错误: {e}")
            finally:
                if client_socket:
                    try:
                        client_socket.close()
                    except:
                        pass

            # 重连延迟
            if self.running:
                time.sleep(2)

# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    # 创建客户端，最大缓存 10 条数据
    client = DataClient(host="127.0.0.1", port=8888, queue_size=10)
    client.key_signals = ["speed", "temperature", "pressure"]  # 关注的信号

    try:
        client.start()

        while True:
            latest = client.get_latest_data()
            if latest:
                print(f"[最新数据] 时间: {latest['timestamp']}, 信号: {latest['signals']}")
            time.sleep(1)  # 每秒打印一次最新数据
    except KeyboardInterrupt:
        print("正在停止客户端...")
        client.stop()