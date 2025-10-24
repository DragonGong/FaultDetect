import time

from monitor.client.visualizer import Visualizer
HOST = "192.168.126.45"
PORT = 12345
if __name__ == "__main__":
    v = Visualizer(host=HOST,port=PORT)
    v.run_realtime_visualization()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭...")
    finally:
        v.client.stop()
        print("程序已退出")