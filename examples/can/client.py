from can_llm.UI_client import UIClient
from can_llm.can_reader import CanReader
from can_llm.utils import READ_REPLAY
from can_llm.utils import Config
import can

dbc_file = '../assets/20250409.dbc'
blf_file = '../assets/logfile/Logging2025-04-09_11-23-34.blf'

bus = can.interface.Bus(
    interface='kvaser',
    channel=0,  # Kvaser 设备通道号
    bitrate=500000  # 波特率，例如 500K
)
host = '127.0.0.1'
port = 65433
signal_name = 'SGW_IBC_PedalTravelSensorSt'
if __name__ == '__main__':
    config_client = Config('config/config.yaml')
    reader = CanReader(mode=READ_REPLAY, dbc_file_path=dbc_file, blf_file_path=blf_file
                       , signal_name=signal_name, bus=bus)
    client = UIClient(config_client=config_client, can_reader=reader)
    client.lanuch_UI()
