import threading
import time
from monitor.server.compute import TrajectoryIntegrator
from monitor.server.server import SocketServer
class DataBroadcaster:
    def __init__(self, integrator: TrajectoryIntegrator, server: SocketServer):
        self.integrator = integrator
        self.server = server
        self.running = False

    def start(self):
        """启动广播线程"""
        if self.running:
            return
        self.running = True
        thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        thread.start()
        print("DataBroadcaster: 启动广播线程")

    def stop(self):
        self.running = False

    def _broadcast_loop(self):
        last_size = 0
        while self.running:
            # 获取最新一条轨迹数据
            with self.integrator.lock:
                if len(self.integrator.trajectory_queue) > 0:
                    latest_data = self.integrator.trajectory_queue[-1]
                else:
                    time.sleep(0.01)
                    continue

            # 广播给所有客户端
            try:
                self.server.broadcast(latest_data)
                # print(f"broad data :{latest_data}")
            except Exception as e:
                print(f"广播异常: {e}")

            # 调试：打印广播频率
            if len(self.integrator.trajectory_queue) != last_size:
                last_size = len(self.integrator.trajectory_queue)
                if last_size % 50 == 0:
                    print(f"已广播 {last_size} 条轨迹数据")

            time.sleep(0.05)  # 控制广播频率（约20Hz）

