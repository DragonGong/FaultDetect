import socket
import gradio as gr
import os
from can_llm.utils import END_OF_ANSWER_BYTE, END_OF_QUESTION, END_OF_FILE_BYTE
from can_llm.can_reader import CanReader
from can_llm.utils import Config


class UIClient:
    def __init__(self, config_client: Config,
                 can_reader: CanReader,
                 ):

        self.host = config_client.ui_client.host
        self.port = config_client.ui_client.port
        self.reader: CanReader = can_reader

    def _update_can_value(self):
        return self.reader.read_can()

    def _query_model(self, user_input):
        current_can = self._update_can_value()

        if not user_input:
            return current_can, "请输入问题"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
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

    def _upload_document(self, file_obj):
        file_path = file_obj.name

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
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

    def _handle_user_input(self, user_input):
        return self._query_model(user_input)

    def _update_display(self):
        current_can = self._update_can_value()
        return current_can, "实时监控中..."

    def lanuch_UI(self):
        with gr.Blocks() as interface:
            with gr.Row():
                can_display = gr.Textbox(label="CAN信号", interactive=False)
                response_display = gr.Textbox(label="诊断结果", interactive=False)

            input_box = gr.Textbox(label="请输入问题")
            submit_btn = gr.Button("提交问题")

            with gr.Row():
                upload_box = gr.File(label="上传知识文档（txt 或 docx）", file_types=[".txt", ".docx"])
                upload_btn = gr.Button("提交文档")

            submit_btn.click(self._handle_user_input, inputs=input_box, outputs=[can_display, response_display])
            upload_btn.click(self._upload_document, inputs=upload_box, outputs=[can_display, response_display])

            interface.load(self._update_display, inputs=None, outputs=[can_display, response_display])

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
            refresh_btn.click(self._update_display, inputs=None, outputs=[can_display, response_display])

            gr.HTML(f"<script>{refresh_js}()</script>")
        interface.launch()
