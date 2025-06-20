import can
import cantools
import time
from can.bus import BusABC
from can_llm.utils import READ_REALTIME, READ_REPLAY
from typing import Literal
import logging


class CanReader:
    def __init__(self,
                 mode: Literal["RT", "RP"] = READ_REALTIME,
                 timeout: int = 3,
                 blf_file_path: str = None,
                 dbc_file_path: str = None,
                 bus: BusABC = None,
                 signal_name: str = 'SGW_IBC_PedalTravelSensorSt'
                 ):
        self.message_cache = None
        self.current_index = None
        self.db = None
        self.mode = mode
        self.bus = bus
        self.signal_name = signal_name
        self.timeout = timeout
        if self.mode == READ_REPLAY:
            if blf_file_path is None:
                logging.error("blf_file_path is none")
            with can.BLFReader(blf_file_path) as blf_messages:
                self.message_cache = list(blf_messages)
                logging.info(f"已缓存 {len(self.message_cache)} 条CAN消息")
                self.current_index = 0
            if dbc_file_path is None:
                logging.error("dbc_file_path is none")
            self.db = cantools.database.load_file(dbc_file_path)

    def current_index_update(self):
        if len(self.message_cache) <= 0:
            logging.error("message_cache is none")
            return
        self.current_index = (self.current_index + 1) % len(self.message_cache)

    def read_can(self):
        if self.mode == READ_REALTIME:
            return self._read_can_RT(self.signal_name, self.timeout)
        else:
            return self._read_can_RP(self.signal_name)

    def _read_can_RT(self, signal_name='SGW_IBC_PedalTravelSensorSt', timeout=3):
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.bus.recv(timeout=1.0)
            if msg is None:
                continue

            try:
                # 根据 ID 自动匹配对应的 CAN 消息定义
                decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                if signal_name in decoded.keys():
                    print("Decoded:", decoded)
                    return decoded

            except Exception as e:
                import traceback
                logging.error(f"read can error:{e}")
                traceback.print_exc()
                return None
        logging.error(f"read timeout :timeout is {timeout} s")
        return None

    def _read_can_RP(self, signal='SGW_IBC_PedalTravelSensorSt'):
        max_retry = len(self.message_cache)
        for _ in range(max_retry):
            self.current_index_update()
            msg = self.message_cache[self.current_index]
            try:
                decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                if signal in decoded:
                    can_value = str(decoded[signal])
                    print(f'arbitration_id:{msg.arbitration_id} is right,can was read')
                    return can_value
            except Exception as e:
                logging.error(f"发生其他错误: {e}")
                import traceback
                traceback.print_exc()
        return "ERROR"
