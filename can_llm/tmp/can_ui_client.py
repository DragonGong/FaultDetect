# # can_ui_client.py
# import can
# import cantools
# import socket
# import threading
# import gradio as gr
# import time

# dbc_file = '20250409.dbc'
# blf_file = 'Logging2025-04-09_11-23-34.blf'

# db = cantools.database.load_file(dbc_file)
# can_value = "N/A"

# messages_cache = []  # 缓存BLF文件中的消息
# current_index = 0    # 当前读取位置

# # 初始化时读取BLF文件并缓存消息
# def init_cache():
#     global messages_cache
#     with can.BLFReader(blf_file) as blf_messages:
#         messages_cache = list(blf_messages)
#     print(f"已缓存 {len(messages_cache)} 条CAN消息")

# # 定期更新CAN值的函数
# def update_can_value():
#     global can_value, current_index, messages_cache

#     if not messages_cache:
#         return "未找到CAN消息"

#     # 从缓存中获取下一条消息
#     msg = messages_cache[current_index]
#     current_index = (current_index + 1) % len(messages_cache)  # 循环读取

#     try:
#         decoded = db.decode_message(msg.arbitration_id, msg.data)
#         if 'SGW_IBC_PedalTravelSensorSt' in decoded:
#             can_value = str(decoded['SGW_IBC_PedalTravelSensorSt'])
#     except Exception:
#         pass

#     return can_value

# # 与模型服务通信
# def query_model(user_input):
#     # 每次查询前更新一次CAN值
#     current_can = update_can_value()

#     if not user_input:
#         return current_can, "请输入问题"

#     HOST = '127.0.0.1'
#     PORT = 65433
#     try:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((HOST, PORT))
#             payload = f"CAN:{current_can}\nUSER:{user_input}"
#             s.sendall(payload.encode('utf-8'))
#             result = s.recv(8192).decode('utf-8')
#         return current_can, result
#     except Exception as e:
#         return current_can, f"连接错误: {e}"

# # 定时更新显示的函数
# def update_display():
#     current_can = update_can_value()
#     return current_can, "实时监控中..."

# # 处理用户输入的函数
# def handle_user_input(user_input):
#     return query_model(user_input)

# # 初始化缓存
# init_cache()

# # 创建一个定时器线程，每秒更新一次显示
# def periodic_update(can_display, response_display):
#     while True:
#         time.sleep(1)  # 每秒更新一次
#         current_can = update_can_value()
#         # 使用queue在线程间传递更新
#         can_display.update(value=current_can)
#         response_display.update(value="实时监控中...")

# # 构建Gradio界面
# with gr.Blocks() as interface:
#     with gr.Row():
#         can_display = gr.Textbox(label="CAN信号", interactive=False)
#         response_display = gr.Textbox(label="诊断结果", interactive=False)

#     input_box = gr.Textbox(label="请输入问题")
#     submit_btn = gr.Button("提交问题")

#     # 设置提交按钮事件
#     submit_btn.click(handle_user_input, inputs=input_box, outputs=[can_display, response_display])

#     # 设置页面加载事件
#     interface.load(update_display, inputs=None, outputs=[can_display, response_display])

#     # 添加JS定时刷新代码
#     refresh_js = """
#     function() {
#         function refresh() {
#             document.querySelector("button.refresh-button").click();
#         }
#         setInterval(refresh, 1000);
#     }
#     """

#     refresh_btn = gr.Button("刷新", visible=False, elem_classes=["refresh-button"])
#     refresh_btn.click(update_display, inputs=None, outputs=[can_display, response_display])

#     # 添加JS代码到页面
#     gr.HTML(f"<script>{refresh_js}()</script>")

# if __name__ == "__main__":
#     interface.launch()
# # # 读取 CAN 报文线程
# # def read_can():
# #     global can_value
# #     with can.BLFReader(blf_file) as blf_messages:
# #         for msg in blf_messages:
# #             try:
# #                 decoded = db.decode_message(msg.arbitration_id, msg.data)
# #                 if 'SGW_IBC_PedalTravelSensorSt' in decoded:
# #                     can_value = str(decoded['SGW_IBC_PedalTravelSensorSt'])
# #             except Exception:
# #                 continue

# # # 启动线程持续读取 CAN
# # threading.Thread(target=read_can, daemon=True).start()

# # # 与模型服务通信
# # def query_model(user_input):
# #     global can_value
# #     HOST = '127.0.0.1'
# #     PORT = 65433
# #     try:
# #         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
# #             s.connect((HOST, PORT))
# #             payload = f"CAN:{can_value}\nUSER:{user_input}"
# #             s.sendall(payload.encode('utf-8'))
# #             result = s.recv(8192).decode('utf-8')
# #         return can_value, result
# #     except Exception as e:
# #         return can_value, f"连接错误: {e}"

# # # 构建 Gradio 界面
# # def update_display(user_input):
# #     return query_model(user_input)

# # can_display = gr.Textbox(label="CAN信号", interactive=False)
# # response_display = gr.Textbox(label="诊断结果", interactive=False)
# # input_box = gr.Textbox(label="请输入问题")

# # interface = gr.Interface(
# #     fn=update_display,
# #     inputs=input_box,
# #     outputs=[can_display, response_display],
# #     live=False
# # )

# # if __name__ == "__main__":
# #     interface.launch()


# can_ui_client.py
import can
import cantools
import socket
import gradio as gr
import os
from can_llm.utils import END_OF_ANSWER_BYTE, END_OF_QUESTION, END_OF_FILE_BYTE

dbc_file = '../doc/20250409.dbc'
blf_file = '../doc/logfile/Logging2025-04-09_11-23-34.blf'

db = cantools.database.load_file(dbc_file)
can_value = "N/A"

messages_cache = []
current_index = 0


def init_cache():
    global messages_cache
    with can.BLFReader(blf_file) as blf_messages:
        messages_cache = list(blf_messages)
    print(f"已缓存 {len(messages_cache)} 条CAN消息")


def update_can_value():
    global can_value, current_index, messages_cache

    if not messages_cache:
        return "未找到CAN消息"

    msg = messages_cache[current_index]
    current_index = (current_index + 1) % len(messages_cache)

    try:
        decoded = db.decode_message(msg.arbitration_id, msg.data)
        if 'SGW_IBC_PedalTravelSensorSt' in decoded:
            can_value = str(decoded['SGW_IBC_PedalTravelSensorSt'])
    except Exception as e:
        print(f"发生其他错误: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR"

    return can_value


def query_model(user_input):
    current_can = update_can_value()

    if not user_input:
        return current_can, "请输入问题"

    HOST = '127.0.0.1'
    PORT = 65433
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            header = f"QUESTION\n"
            s.sendall(header.encode('utf-8'))

            ack = s.recv(1024)
            print("head ack from server")

            payload = f"CAN:{current_can}\nUSER:{user_input}"
            payload += END_OF_QUESTION
            s.sendall(payload.encode('utf-8'))
            result_data = b''
            while END_OF_ANSWER_BYTE not in result_data:
                result_data += s.recv(1024)
            result_data = result_data.replace(END_OF_ANSWER_BYTE, b"")
            result = result_data.decode('utf-8')
            print(f"recv:{result}")
        return current_can, result
    except Exception as e:
        return current_can, f"连接错误: {e}"


def upload_document(file_obj):
    file_path = file_obj.name
    HOST = '127.0.0.1'
    PORT = 65433
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            header = f"UPLOAD:{os.path.basename(file_path)}\n"
            s.sendall(header.encode('utf-8'))

            ack = s.recv(1024)
            print("head ack from server")
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk:
                        break
                    s.sendall(chunk)

            s.sendall(END_OF_FILE_BYTE)
            result = s.recv(4096).decode('utf-8')
            print(f"recv:{result}")
            return "CAN信号未更新", result
    except Exception as e:
        return "N/A", f"文件上传失败: {e}"


def update_display():
    current_can = update_can_value()
    return current_can, "实时监控中..."


def handle_user_input(user_input):
    return query_model(user_input)


init_cache()

with gr.Blocks() as interface:
    with gr.Row():
        can_display = gr.Textbox(label="CAN信号", interactive=False)
        response_display = gr.Textbox(label="诊断结果", interactive=False)

    input_box = gr.Textbox(label="请输入问题")
    submit_btn = gr.Button("提交问题")

    with gr.Row():
        upload_box = gr.File(label="上传知识文档（txt 或 docx）", file_types=[".txt", ".docx"])
        upload_btn = gr.Button("提交文档")

    submit_btn.click(handle_user_input, inputs=input_box, outputs=[can_display, response_display])
    upload_btn.click(upload_document, inputs=upload_box, outputs=[can_display, response_display])

    interface.load(update_display, inputs=None, outputs=[can_display, response_display])

    # 当前是每隔一秒刷新，后续可能需要修改具体逻辑
    refresh_js = """
    function() {
        function refresh() {
            document.querySelector("button.refresh-button").click();
        }
        setInterval(refresh, 1000);
    }
    """

    refresh_btn = gr.Button("刷新", visible=False, elem_classes=["refresh-button"])
    refresh_btn.click(update_display, inputs=None, outputs=[can_display, response_display])

    gr.HTML(f"<script>{refresh_js}()</script>")

if __name__ == "__main__":
    interface.launch()
