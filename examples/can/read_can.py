import can
import cantools

# Step 1: 加载 DBC 文件

dbc_file = 'assets/20250409.dbc'
db = cantools.database.load_file(dbc_file)
# # 初始化 Kvaser CAN 通道
bus = can.interface.Bus(
    interface='kvaser',
    channel=0,             # 你的 Kvaser 设备通道号
    bitrate=500000         # 波特率，例如 500K
)

print("Listening on CAN bus...")

while True:
    msg = bus.recv(timeout=1.0)
    if msg is None:
        continue

    try:
        # 根据 ID 自动匹配对应的 CAN 消息定义
        decoded = db.decode_message(msg.arbitration_id, msg.data)
        if ('SGW_IBC_PedalTravelSensorSt' in decoded.keys()):
            # print(f"Message ID: {hex(msg.arbitration_id)}")
            print("Decoded:", decoded)
        
    except Exception as e:
        continue
        # print(f"Failed to decode message: {e}")


