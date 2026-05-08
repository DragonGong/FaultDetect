# FaultDetect

一个面向车辆故障检测与 CAN 数据处理的实验项目，包含：

- **CAN 数据采集与解析**（基于 `python-can` + `cantools`）；
- **实时数据服务与广播**（Socket 服务端/客户端）；
- **轨迹与速度可视化**（Matplotlib 动画）；
- **故障诊断原型**（`can_llm` 目录中的知识库/模型服务实验代码）。

> 当前仓库以研究与原型验证为主，模块之间相对独立，可按需单独运行。

---

## 目录说明

```text
FaultDetect/
├── monitor/                 # 实时监控主模块（server + client）
│   ├── server/              # 读取 CAN、计算状态、Socket 广播
│   └── client/              # Socket 客户端与可视化
├── can_llm/                 # 故障诊断与知识图谱相关实验代码
├── examples/                # 示例脚本（CAN、monitor、vision、demos）
├── config/                  # 配置样例
├── requirements.txt         # Python 依赖
└── README.md
```

---

## 环境要求

- Python **3.9+**（推荐 3.10）
- Linux / macOS / Windows
- 可选：CAN 硬件设备与对应驱动（若只跑示例可不接硬件）

---

## 安装

在项目根目录执行：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```

如需以“可编辑模式”开发（修改源码立即生效）：

```bash
pip install -e .
```

---

## 快速开始

### 1) 启动实时数据服务端（monitor）

```bash
python -m monitor.server.server
```

默认监听地址通常为 `127.0.0.1:12345`（以代码参数为准）。

### 2) 启动可视化客户端

```bash
python -m monitor.client.visualizer
```

客户端会连接服务端，实时显示轨迹、速度和姿态信息。

---

## 常见运行入口

你可以按场景直接运行 `examples/` 下脚本：

- `examples/monitor/`：监控链路测试；
- `examples/can/`：CAN 收发与解析示例；
- `examples/demos/`：socket 与小工具 demo；
- `examples/vision/`：视觉模型训练/验证脚本。

示例（按实际环境调整）：

```bash
python examples/can/read_can.py
python examples/monitor/test_car.py
```

---

## 配置说明

- 示例配置文件位于：`config/config-example.yaml`。
- 建议复制为你自己的本地配置后再修改：

```bash
cp config/config-example.yaml config/config.yaml
```

---

## 开发建议

- 新增模块时，优先放在 `monitor/` 或 `examples/` 对应子目录；
- 保持“采集 -> 计算 -> 广播 -> 可视化”链路解耦，便于替换数据源；
- 与硬件强相关的参数（通道、bitrate、dbc 路径）统一放配置文件。

---

## 故障排查

1. **客户端连不上服务端**
   - 检查服务端是否已启动；
   - 检查 IP/端口是否一致；
   - 检查本机防火墙是否拦截。

2. **CAN 数据为空或解析失败**
   - 检查 CAN 硬件与驱动；
   - 检查总线参数（如 bitrate）；
   - 检查 DBC 文件路径及信号名是否匹配。

3. **可视化窗口无刷新**
   - 确认客户端收到数据；
   - 降低刷新频率或减少绘图负载；
   - 在无图形界面环境下使用本地桌面或转存日志。

---

## License

如需开源发布，建议在仓库根目录补充 `LICENSE` 文件并在此处声明。
