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
from langchain.chains import GraphCypherQAChain
from langchain import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from check_update_kg import check_kg, update_kg
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY, OPENAI_API_BASE

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
                    data = conn.recv(4096)
                    if not data:
                        continue

                    decoded = data.decode('utf-8', errors='ignore')
                    if decoded.startswith("UPLOAD:"):
                        filename = decoded.split(":", 1)[1].strip()
                        update.process_document(filename, graph)
                        conn.sendall("知识库文件处理完成，已更新至图谱。".encode('utf-8'))
                        continue 
                        # save_path = f"uploaded_{filename}"

                        # with open(save_path, "wb") as f:
                        #     while True:
                        #         chunk = conn.recv(4096)
                        #         if b"<<END_OF_FILE>>" in chunk:
                        #             chunk = chunk.replace(b"<<END_OF_FILE>>", b"")
                        #             f.write(chunk)
                        #             break
                        #         f.write(chunk)
                        
                        # try:
                        #     # 确保文件存在且可读
                        #     if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                        #         # 添加调试信息
                        #         print(f"处理文件: {save_path}")
                        #         print(f"文件大小: {os.path.getsize(save_path)} 字节")
                                
                        #         # 确保文件完全写入磁盘
                        #         import time
                        #         time.sleep(0.5)  # 给文件系统一点时间
                                
                        #         # 将相对路径转为绝对路径
                        #         abs_path = os.path.abspath(save_path)
                        #         print(f"使用绝对路径: {save_path}")
                        #         print(f"type: {type(abs_path)}")
                        #         # save_path = 'demo/dmo.txt'
                                
                        #         # 使用绝对路径调用处理方法
                        #         update.process_document(abs_path, graph)
                        #         conn.sendall("知识库文件处理完成，已更新至图谱。".encode('utf-8'))
                        #     else:
                        #         conn.sendall("文件上传失败或为空文件。".encode('utf-8'))
                        # except Exception as e:
                        #     import traceback
                        #     error_details = traceback.format_exc()
                        #     print(f"文件处理错误详情: {error_details}")
                        #     conn.sendall(f"处理文件时出错: {str(e)}".encode('utf-8'))
                        # finally:
                        #     # 无论成功与否都清理文件
                        #     if os.path.exists(save_path):
                        #         continue
                        #         # os.remove(save_path)
                        # continue

            
                    
                    # 正常问题问答处理
                    parts = decoded.strip().split('\n')
                    can_data = parts[0].replace("CAN:", "").strip()
                    user_input = parts[1].replace("USER:", "").strip()

                    result = handle_request(user_input, can_data)
                    conn.sendall(result.encode('utf-8'))

                except Exception as e:
                    conn.sendall(f"处理出错: {str(e)}".encode('utf-8'))
