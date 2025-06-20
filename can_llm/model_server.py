import os
import socket
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

from can_llm.check_update_kg import check_kg, update_kg
from can_llm.utils import END_OF_FILE_BYTE, END_OF_QUESTION_BYTE, END_OF_ANSWER, END_OF_QUESTION, ACK_BYTE
from can_llm.utils import Config


class ModelServer:
    def __init__(self, config: Config):
        os.environ["NEO4J_URI"] = config.neo4j.uri
        os.environ["NEO4J_USERNAME"] = config.neo4j.username
        os.environ["NEO4J_PASSWORD"] = config.neo4j.password
        os.environ["OPENAI_API_KEY"] = config.openai.api_key
        os.environ["OPENAI_API_BASE"] = config.openai.api_base

        self.check = check_kg()
        self.update = update_kg()
        self.graph = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"]
        )
        self.lm = ChatOpenAI(temperature=0, model_name="gpt-4o")

        self.HOST = config.model_server.host
        self.PORT = config.model_server.port

    def handle_request(self, user_input, can_data):
        combined_input = f"{user_input}\nCAN data:{can_data}"
        check_response = self.check.transfer_prompt(user_input=combined_input, graph=self.graph)

        if "don't know" in check_response or "don't have" in check_response:
            system_prompt = "You are a car maintenance engineer, now the customer will describe to you his car related problems, from the perspective of the internal structure of the car from high to low probability, what is the possible problem of what parts of the car."
            llm_response = self.update.llm(system_prompt + "user_input:" + user_input + "\nCAN Data:" + can_data)
            self.update.update_graph(llm_response.content, self.graph)
            return llm_response.content
        else:
            final_result = self.lm(f"user_input:{user_input}\nresponse:{check_response}\n请用中文简洁分析")
            return final_result.content

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.HOST, self.PORT))
            s.listen()
            print(f"[大模型服务] 正在监听 {self.HOST}:{self.PORT} ...")

            while True:
                conn, addr = s.accept()
                with conn:
                    try:
                        header = b""
                        while b"\n" not in header:
                            header += conn.recv(1024)
                        header_str = header.decode('utf-8').strip()
                        print("header :" + header_str)
                        conn.sendall(ACK_BYTE)
                        if header_str.startswith("UPLOAD"):
                            filename = header_str.split(':')[1]
                            file_data = b''
                            while END_OF_FILE_BYTE not in file_data:
                                file_data += conn.recv(1024)
                            file_data = file_data.replace(END_OF_FILE_BYTE, b'')

                            self.update.process_document(filename, str(file_data.decode('utf-8')), self.graph)
                            conn.sendall("知识库文件处理完成，已更新至图谱。".encode('utf-8'))
                            print(f"send: 知识库文件处理完成，已更新至图谱。")

                        else:
                            question_data = b''
                            while END_OF_QUESTION_BYTE not in question_data:
                                question_data += conn.recv(1024)
                            parts = question_data.decode('utf-8').strip().split('\n')
                            can_data = parts[0].replace("CAN:", "").strip()
                            user_input = parts[1].replace("USER:", "").strip()
                            user_input = user_input.replace(END_OF_QUESTION, "")
                            result = str(self.handle_request(user_input, can_data)) + END_OF_ANSWER
                            conn.sendall(result.encode('utf-8'))
                            print(f"send: {result}")
                    except Exception as e:
                        print(f"发生其他错误: {e}")
                        import traceback
                        traceback.print_exc()
