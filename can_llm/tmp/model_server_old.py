# # model_server.py
# import os
# import socket
# from langchain.chains import GraphCypherQAChain
# from langchain import PromptTemplate
# from langchain_community.graphs import Neo4jGraph
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_openai import ChatOpenAI
# from check_update_kg import check_kg, update_kg

# # 环境配置
# os.environ["NEO4J_URI"] = "neo4j+s://5fb1efdc.databases.neo4j.io"
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "nJXKEQsPrratzUc_KppxqivlVPmB6DcjsbVIfXC0dpk"
# os.environ["OPENAI_API_KEY"] = "sk-BXYyzivxRAF2SpSh8eA84203B01d4aF8Af87F339E48b8a31"
# os.environ["OPENAI_API_BASE"] = "https://ai.pumpkinai.online/v1"

# # 实例化模块
# check = check_kg()
# update = update_kg()
# graph = Neo4jGraph(
#     url=os.environ["NEO4J_URI"],
#     username=os.environ["NEO4J_USERNAME"],
#     password=os.environ["NEO4J_PASSWORD"]
# )
# lm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# # 服务端 socket 逻辑
# HOST = '127.0.0.1'
# PORT = 65433

# def handle_request(user_input, can_data):
#     combined_input = f"{user_input}\nCAN信号:{can_data}"
#     check_response = check.transfer_prompt(user_input=combined_input, graph=graph)

#     if "don't know" in check_response or "don't have" in check_response:
#         system_prompt = "你是一位汽车维修工程师，请根据以下描述推测可能的故障部位："
#         llm_response = update.llm(system_prompt + "用户描述:" + user_input + "\nCAN数据:" + can_data)
#         update.update_graph(llm_response.content, graph)
#         return llm_response.content
#     else:
#         final_result = lm(f"user_input:{user_input}\nresponse:{check_response}\n请用中文简洁分析")
#         return final_result.content

# # 启动 socket 服务
# if __name__ == "__main__":
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind((HOST, PORT))
#         s.listen()
#         print(f"[大模型服务] 正在监听 {HOST}:{PORT} ...")

#         while True:
#             conn, addr = s.accept()
#             with conn:
#                 data = conn.recv(4096)
#                 if not data:
#                     continue
#                 try:
#                     decoded = data.decode('utf-8')
#                     parts = decoded.strip().split('\n')
#                     can_data = parts[0].replace("CAN:", "").strip()
#                     user_input = parts[1].replace("USER:", "").strip()

#                     result = handle_request(user_input, can_data)
#                     conn.sendall(result.encode('utf-8'))
#                 except Exception as e:
#                     conn.sendall(f"处理出错: {str(e)}".encode('utf-8'))


import os
import socket
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

from can_llm import utils
from check_update_kg import check_kg, update_kg
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY, OPENAI_API_BASE
from can_llm.utils import END_OF_FILE_BYTE, END_OF_QUESTION_BYTE, END_OF_ANSWER, END_OF_QUESTION

# 直接使用这些变量
# 环境配置
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

# 实例化模块
check = check_kg()
update = update_kg()
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)
lm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# 服务端配置
HOST = '127.0.0.1'
PORT = 65433

# 请求处理逻辑
def handle_request(user_input, can_data):
    combined_input = f"{user_input}\nCAN data:{can_data}"
    check_response = check.transfer_prompt(user_input=combined_input, graph=graph)

    if "don't know" in check_response or "don't have" in check_response:
        system_prompt = "You are a car maintenance engineer, now the customer will describe to you his car related problems, from the perspective of the internal structure of the car from high to low probability, what is the possible problem of what parts of the car."
        llm_response = update.llm(system_prompt + "user_input:" + user_input + "\nCAN Data:" + can_data)
        update.update_graph(llm_response.content, graph)
        return llm_response.content
    else:
        final_result = lm(f"user_input:{user_input}\nresponse:{check_response}\n请用中文简洁分析")
        return final_result.content

# 启动 socket 服务
if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"[大模型服务] 正在监听 {HOST}:{PORT} ...")

        while True:
            conn, addr = s.accept()
            with conn:
                try:
                    header = b""
                    while b"\n" not in header:
                        header += conn.recv(1024)
                    header_str =  header.decode('utf-8').strip()
                    print("header :"+header_str)
                    conn.sendall(utils.ACK_BYTE)
                    if header_str.startswith("UPLOAD"):
                        filename = header_str.split(':')[1]
                        file_data = b''
                        while END_OF_FILE_BYTE not in file_data:
                            file_data += conn.recv(1024)
                        file_data = file_data.replace(END_OF_FILE_BYTE,b'')

                        update.process_document(filename ,str(file_data.decode('utf-8')) , graph)
                        conn.sendall("知识库文件处理完成，已更新至图谱。".encode('utf-8'))
                        print(f"send: 知识库文件处理完成，已更新至图谱。")

                    else:
                        question_data = b''
                        while END_OF_QUESTION_BYTE not in question_data:
                            question_data += conn.recv(1024)
                        parts = question_data.decode('utf-8').strip().split('\n')
                        can_data = parts[0].replace("CAN:", "").strip()
                        user_input = parts[1].replace("USER:", "").strip()
                        user_input = user_input.replace(END_OF_QUESTION,"")
                        result = str(handle_request(user_input, can_data))+END_OF_ANSWER
                        conn.sendall(result.encode('utf-8'))
                        print(f"send: {result}")
                except Exception as e:
                    print(f"发生其他错误: {e}")
                    import traceback
                    traceback.print_exc()

