import time

from monitor.server.broad_caster import DataBroadcaster
from monitor.server.can_reader import CANReader
from monitor.server.compute import TrajectoryIntegrator
from monitor.server.server import SocketServer

DBC_FILE = r'.\assets\监测信号+20251024(2).dbc'
CAN_CHANNEL = 0
BITRATE = 500000
TARGET_SIGNAL = [
    'SGW_IVI_GyroX', 'SGW_IVI_GyroY', 'SGW_IVI_GyroZ',
    'SGW_IVI_AccelX', 'SGW_IVI_AcceY', 'SGW_IVI_AccelZ',
    'ACU_YawRateSt', 'IBC_VehicleSpeed', 'TAS_SAS_SteeringAngle','PDCU_ActualGear'
]
MAX_QUEUE_SIZE = 100
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 12345


if __name__ == '__main__':
    can_reader = CANReader(dbc_file=DBC_FILE,channel=CAN_CHANNEL,bitrate=BITRATE,target_signals=TARGET_SIGNAL)
    can_reader.start_thread()
    integrator=TrajectoryIntegrator(can_reader=can_reader)
    integrator.start()
    server = SocketServer(host=SERVER_HOST,port=SERVER_PORT)
    server.start()
    bc = DataBroadcaster(integrator,server)
    bc.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
    finally:
        can_reader.stop_thread()
        integrator.stop()
        bc.stop()
        server.stop()
        print("程序已退出")