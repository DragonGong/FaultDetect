#!/usr/bin/env python3
import socket
import time
import sys
import can
import cantools
import json
import threading
from queue import Queue
import signal

# ========== 配置参数 ==========
DBC_FILE = r'E:\code\FaultDetect\assets\20250626.dbc'
CAN_CHANNEL = 0
BITRATE = 500000
TARGET_SIGNAL = [
    'SGW_IVI_GyroX', 'SGW_IVI_GyroY', 'SGW_IVI_GyroZ',
    'SGW_IVI_AccelX', 'SGW_IVI_AcceY', 'SGW_IVI_AccelZ',
    'ACU_YawRateSt', 'IBC_VehicleSpeed', 'TAS_SAS_SteeringAngle'
]
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 12345
MAX_QUEUE_SIZE = 100  # 消息队列最大长度
# ==============================

# 全局变量
can_queue = Queue(maxsize=MAX_QUEUE_SIZE)
running = True
bus = None
db = None


def can_reader():
    """CAN信号读取线程，负责从CAN总线接收数据并放入队列"""
    global bus, db, running

    try:
        while running:
            if bus is None:
                time.sleep(0.1)
                continue

            # 接收CAN消息
            msg = bus.recv(timeout=0.5)
            if msg is None:
                continue

            try:
                # 解码消息
                decoded = db.decode_message(msg.arbitration_id, msg.data)
                # print(decoded)
                # 检查是否包含目标信号
                if any(signal in decoded for signal in TARGET_SIGNAL):
                    # 构造要发送的数据结构
                    data_to_send = {
                        "timestamp": time.time(),
                        "can_id": hex(msg.arbitration_id),
                        "signals": decoded
                    }

                    # 放入队列，如果队列满则丢弃 oldest 消息
                    if can_queue.full():
                        try:
                            can_queue.get_nowait()  # 移除最旧的消息
                        except:
                            pass
                    can_queue.put(data_to_send)
                    # print(data_to_send)
            except Exception as e:
                print(f"CAN消息解码错误: {e}")
                continue

    except Exception as e:
        print(f"CAN读取线程错误: {e}")
    finally:
        print("CAN读取线程已退出")


def client_handler(client_socket):
    """处理客户端连接的函数，从队列中获取数据并发送"""
    global running

    try:
        print("开始向客户端发送数据...")
        while running:
            try:
                # 从队列获取数据，超时退出避免阻塞
                data = can_queue.get(timeout=1)
                can_queue.task_done()

                # 序列化为JSON并发送
                message = json.dumps(data) + "\n"
                if ('TAS_SAS_SteeringAngle') in message:
                    print(message)
                client_socket.send(message.encode('utf-8'))

            except Exception as e:
                # 超时属于正常情况，继续循环
                if "timeout" not in str(e).lower():
                    print(f"发送数据错误: {e}")

    except ConnectionResetError:
        print("客户端已强制断开连接")
    except Exception as e:
        print(f"客户端处理错误: {e}")
    finally:
        client_socket.close()
        print("客户端连接已关闭")


def socket_server():
    """Socket服务器线程，负责接受客户端连接"""
    global running

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen(5)
        server_socket.settimeout(1)  # 设置超时，便于退出循环
        print(f"服务器已启动，监听 {SERVER_HOST}:{SERVER_PORT}")

        while running:
            try:
                client_socket, addr = server_socket.accept()
                print(f"新客户端连接: {addr}")

                # 为每个客户端创建一个处理线程
                client_thread = threading.Thread(
                    target=client_handler,
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()

            except socket.timeout:
                continue  # 超时，继续循环检查running状态
            except Exception as e:
                print(f"服务器错误: {e}")
                break

    except Exception as e:
        print(f"服务器启动失败: {e}")
    finally:
        server_socket.close()
        print("服务器已关闭")


def signal_handler(sig, frame):
    """处理程序退出信号"""
    global running
    print("\n接收到退出信号，正在关闭程序...")
    running = False


def main():
    global bus, db, running

    # 注册信号处理器，处理Ctrl+C退出
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 加载DBC文件
        print(f"加载DBC文件: {DBC_FILE}")
        db = cantools.database.load_file(DBC_FILE)

        # 初始化CAN总线
        print(f"初始化CAN总线 (通道: {CAN_CHANNEL}, 波特率: {BITRATE})")
        bus = can.interface.Bus(
            interface='kvaser',
            channel=CAN_CHANNEL,
            bitrate=BITRATE
        )

        # 创建并启动CAN读取线程
        can_thread = threading.Thread(target=can_reader, daemon=True)
        can_thread.start()

        # 启动Socket服务器线程
        server_thread = threading.Thread(target=socket_server, daemon=True)
        server_thread.start()

        # 主线程等待退出信号
        while running:
            time.sleep(1)

    except Exception as e:
        print(f"初始化失败: {e}")
        running = False
    finally:
        # 清理资源
        if bus is not None:
            bus.shutdown()
        print("程序已退出")


if __name__ == "__main__":
    main()
