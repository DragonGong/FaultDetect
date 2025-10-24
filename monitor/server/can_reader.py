import time
import threading
from enum import Enum
from queue import Queue
import can
import cantools
from typing import Dict, Any, Optional, List
from collections import defaultdict


class ReadMode(Enum):
    READ_ONLY = 0
    READ_REPLAY = 1


class CANReader:
    def __init__(self, dbc_file: str , channel: int = 0, bitrate: int = 500000, target_signals=None):
        """
        初始化CAN读取器
        :param dbc_file: DBC文件路径
        :param channel: CAN通道
        :param bitrate: 波特率
        """
        self.dbc_file = dbc_file
        self.channel = channel
        self.bitrate = bitrate
        self.db = None
        self.bus = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.mode = ReadMode.READ_ONLY
        if target_signals is not None:
            self.target_signals = target_signals
        else:
            self.target_signals =  [
                'SGW_IVI_GyroX', 'SGW_IVI_GyroY', 'SGW_IVI_GyroZ',
                'SGW_IVI_AccelX', 'SGW_IVI_AcceY', 'SGW_IVI_AccelZ',
                'ACU_YawRateSt', 'IBC_VehicleSpeed', 'TAS_SAS_SteeringAngle','PDCU_ActualGear'
            ]
            # 加载DBC文件
        try:
            self.db = cantools.database.load_file(dbc_file)
            print(f"DBC文件加载成功: {dbc_file}")
        except Exception as e:
            print(f"DBC文件加载失败: {e}")
            raise

        self.latest_data = {
            "timestamp": None,
            "signals": {sig: 0.0 for sig in self.target_signals}
        }

        self.can_queue = Queue(maxsize=100)  # 默认最大100条，可后续通过方法设置

    def set_queue_size(self, size: int):
        """动态设置队列大小（线程模式用）"""
        self.can_queue = Queue(maxsize=size)

    def start_thread(self):
        """
        启动后台线程持续读取CAN数据
        """
        if self.running:
            print("CAN读取线程已在运行中...")
            return

        # 初始化CAN总线
        try:
            self.bus = can.interface.Bus(
                channel=self.channel,
                bustype='vector',
                bitrate=self.bitrate,
                receive_own_messages=False
            )
            print(f"CAN总线初始化成功: channel={self.channel}, bitrate={self.bitrate}")
        except Exception as e:
            print(f"CAN总线初始化失败: {e}")
            raise

        self.running = True
        self._thread = threading.Thread(target=self._can_reader_loop, args=(), daemon=True)
        self._thread.start()
        print("CAN读取线程已启动")

    def stop_thread(self):
        """停止后台线程"""
        if not self.running:
            return
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.bus:
            self.bus.shutdown()
            self.bus = None
        print("CAN读取线程已停止")

    def _can_reader_loop(self):
        """后台线程主循环：持续读取CAN消息，更新 latest_data"""
        while self.running:
            try:
                if self.bus is None:
                    time.sleep(0.1)
                    continue

                msg = self.bus.recv(timeout=0.5)
                if msg is None:
                    continue

                try:
                    decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                    # 检查是否有目标信号
                    has_target = False
                    for name, value in decoded.items():
                        if name in self.target_signals:
                            self.latest_data["signals"][name] = value
                            has_target = True

                    if has_target:
                        # 只有在更新了目标信号时才更新时间戳
                        self.latest_data["timestamp"] = msg.timestamp # 使用系统时间，确保单调

                except Exception as e:
                    # 解码失败，跳过
                    continue

            except Exception as e:
                print(f"CAN读取线程错误: {e}")
                break

        print("CAN读取线程退出")

    def load_blf_data(self, blf_file: str, key_signals: List[str] = None) -> Dict[str, list]:
        """
        加载并预处理BLF数据文件
        :param blf_file: .blf 文件路径
        :param key_signals: 要提取的关键信号列表
        :return: 解析后的数据字典，格式: {'timestamp': [...], 'SGW_IVI_GyroX': [...], ...}
        """
        data = defaultdict(list)

        try:
            with can.BLFReader(blf_file) as blf_messages:
                for msg in blf_messages:
                    try:
                        decoded_signals = self.db.decode_message(msg.arbitration_id, msg.data)
                        data['timestamp'].append(msg.timestamp)  # 使用CAN消息时间戳

                        for name, value in decoded_signals.items():
                            if name in key_signals:
                                data[name].append(value)
                    except Exception as e:
                        # 忽略无法解码的消息（如未在DBC中定义）
                        continue
        except Exception as e:
            print(f"读取BLF文件失败: {e}")
            raise

        print(f"BLF解析完成，共处理 {len(data['timestamp'])} 条有效消息")
        return dict(data)  # 转为普通 dict

    def get_latest_data(self) -> dict:
        """
        外部调用此函数获取最新数据
        返回一个深拷贝，避免外部修改
        """
        from copy import copy, deepcopy

        # 如果还没收到任何数据
        if self.latest_data["timestamp"] is None:
            return None

        # 返回副本，防止外部修改影响内部状态
        return {
            "timestamp": self.latest_data["timestamp"],
            "signals": deepcopy(self.latest_data["signals"])  # 信号字典也复制
        }

    def read_once(self, target_signals, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        单次读取并解析一条CAN消息（不使用队列）
        :param target_signals: 目标的key
        :param timeout: 接收超时时间
        :return: 解析后的数据字典，格式: {"timestamp": ..., "can_id": "...", "signals": {...}}
                 如果超时或出错，返回 None
        """
        if self.mode == ReadMode.READ_ONLY:
            if self.bus is None:
                try:
                    self.bus = can.interface.Bus(
                        channel=self.channel,
                        bustype='vector',
                        bitrate=self.bitrate,
                        receive_own_messages=False
                    )
                    print(f"临时CAN总线已创建: channel={self.channel}")
                except Exception as e:
                    print(f"创建临时CAN总线失败: {e}")
                    return None

            try:
                msg = self.bus.recv(timeout=timeout)
                if msg is None:
                    return None

                decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                if target_signals is None:
                    return {
                        "timestamp": msg.timestamp,
                        "can_id": hex(msg.arbitration_id),
                        "signals": decoded
                    }
                elif any(signal in decoded for signal in target_signals):
                    return {
                        "timestamp": msg.timestamp,
                        "can_id": hex(msg.arbitration_id),
                        "signals": decoded
                    }
                else:
                    return None
            except Exception as e:
                print(f"单次CAN读取失败: {e}")
                return None
        elif self.mode == ReadMode.READ_REPLAY:
            last_data = self.latest_data.copy()
            if target_signals is None:
                return last_data
            elif any(signal in last_data["signals"] for signal in target_signals):
                return last_data
            else:
                return None
        else:
            return None

    def start_playback(self, blf_file: str, playback_speed: float = 1.0):
        """
        启动BLF文件回放，模拟实时CAN读取
        :param blf_file: .blf 文件路径
        :param playback_speed: 回放速度倍率，1.0为原速，2.0为两倍速，0.5为半速
        """
        self.mode = ReadMode.READ_REPLAY

        def playback_thread():
            try:
                with can.BLFReader(blf_file) as blf_messages:
                    print("blf file was read")
                    messages = list(blf_messages)
                    if not messages:
                        print("BLF文件为空")
                        return

                    # 按时间排序
                    # messages.sort(key=lambda m: m.timestamp)
                    last_time = None

                    for msg in messages:
                        if not self.running:
                            break

                        current_time = msg.timestamp

                        # 计算延迟（基于原始时间间隔）
                        if last_time is not None:
                            delta_t = (current_time - last_time) / playback_speed
                            if delta_t > 0:
                                precise_sleep(delta_t)


                        try:
                            decoded_signals = self.db.decode_message(msg.arbitration_id, msg.data)
                            # print(f"decode signals is {decoded_signals}")

                            # 只更新目标信号
                            for name, value in decoded_signals.items():
                                if name in self.target_signals:
                                    self.latest_data["signals"][name] = value
                                    updated = True
                                    # print("update")
                            self.latest_data["timestamp"] = current_time  # 使用BLF原始时间戳

                        except Exception as e:
                            # 解码失败，跳过
                            # print(f"解码失败：{type(e).__name__} - {e}")
                            pass

                        last_time = current_time

                    print("BLF回放完成")

            except Exception as e:
                print(f"BLF回放出错: {e}")

        # 设置运行标志
        self.running = True
        thread = threading.Thread(target=playback_thread, daemon=True)
        thread.start()
        print(f"BLF回放已启动: {blf_file} (速度: {playback_speed}x)")

    def close(self):
        """关闭总线资源"""
        if self.bus:
            self.bus.shutdown()
            self.bus = None
        print("CAN总线已关闭")


def precise_sleep(seconds):
    if seconds <= 0:
        return
    start = time.perf_counter()
    # 先 sleep 大部分时间
    if seconds > 0.005:  # 大于5ms，先 sleep 掉大部分
        time.sleep(seconds - 0.005)

    # 再用 busy-wait 精确等待剩余时间
    while time.perf_counter() - start < seconds:
        pass  # 空转，直到时间到
# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    # ========== 配置参数 ==========
    DBC_FILE = r'E:\code\FaultDetect\assets\20250409.dbc'
    CAN_CHANNEL = 0
    BITRATE = 500000
    TARGET_SIGNAL = [
        'SGW_IVI_GyroX', 'SGW_IVI_GyroY', 'SGW_IVI_GyroZ',
        'SGW_IVI_AccelX', 'SGW_IVI_AcceY', 'SGW_IVI_AccelZ',
        'ACU_YawRateSt', 'IBC_VehicleSpeed', 'TAS_SAS_SteeringAngle'
    ]
    MAX_QUEUE_SIZE = 100
    BLF_FILE = r"E:\code\FaultDetect\assets\Logging2025-04-09_11-23-34.blf"
    # ==============================

    # # === 模式1：启动线程读取（后台持续运行）===
    # reader = CANReader(dbc_file=DBC_FILE, channel=CAN_CHANNEL, bitrate=BITRATE)
    # reader.set_queue_size(MAX_QUEUE_SIZE)
    # reader.start_thread(TARGET_SIGNAL)
    #
    # try:
    #     # 模拟主程序运行，从队列取数据
    #     while True:
    #         if not reader.can_queue.empty():
    #             data = reader.can_queue.get()
    #             print("【队列数据】", data["timestamp"], data["can_id"], data["signals"].keys())
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     print("停止线程...")
    #     reader.stop_thread()

    # === 模式2：直接读取（单次调用，不启动线程）===
    # reader = CANReader(dbc_file=DBC_FILE, channel=CAN_CHANNEL, bitrate=BITRATE)
    # for i in range(5):
    #     data = reader.read_once(timeout=2.0)
    #     if data:
    #         print(f"【单次读取】{i+1}: {data['timestamp']}, ID={data['can_id']}, Signals={list(data['signals'].keys())}")
    #     else:
    #         print(f"【单次读取】{i+1}: 无数据")
    #     time.sleep(1)
    # reader.close()

    # === 模式3：启动线程，读取blf文件，模拟真实读取 ===
    import time

    reader = CANReader(DBC_FILE)
    reader.start_playback(BLF_FILE, playback_speed=1.0)

    # 模拟外部循环不断读取最新数据
    while True:
        latest = reader.latest_data  # 直接读属性！
        if latest:  # 非空
            # print(latest)
            gyro_z = latest['signals'].get('TAS_SAS_SteeringAngle')
            speed = latest['signals'].get('IBC_VehicleSpeed')
            # if speed is not None or gyro_z is not None:
            #     print(f"最新车速: {speed}, 横摆角速度: {gyro_z}")
        time.sleep(0.2)  # 10ms 查询一次

