from langchain.chains import GraphCypherQAChain
from langchain import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredWordDocumentLoader
from langchain.schema import SystemMessage, HumanMessage
class check_kg:
    def __init__(self):
        self.cypher_qa_prompt = """You are an assistant that helps to form nice and human understandable answers.
        First, identify the key concepts in the question and list their possible synonyms or related terms.
        Then, use these terms to search in the knowledge graph.
        
        For example:
        - If user asks about "engine noise", also consider "motor sound", "engine sound", "unusual noise"
        - If user mentions "won't start", also consider "cannot start", "failing to start", "startup issue"
        - If user mentions "check engine light", also consider "warning light", "indicator light", "MIL"
        - If user mentions "transmission", also consider "gearbox", "gear system", "drive train"
        
        The information part contains the provided information that you must use to construct an answer.
        The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
        Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
        If the provided information is empty, say that you don't know the answer.
        Final answer should be easily readable and structured.
        
        Information:
        {context}

        Question: {question}
        Helpful Answer:"""

    def transfer_prompt(self, user_input, graph):
        if "Battery Voltage" in user_input or "System Voltage" in user_input:
            return self.query_voltage_info(user_input, graph)
        
        qa_prompt = PromptTemplate(input_variables=["context", "question"], template=self.cypher_qa_prompt)
        graph.refresh_schema()
        cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatOpenAI(temperature=0, model_name='gpt-4o'),
            qa_llm=ChatOpenAI(temperature=0),
            graph=graph,
            qa_prompt=qa_prompt,
            verbose=True,
        )
        try:
            response = cypher_chain.run(user_input)
            if not response:
                response = "I don't know the answer."
            return response
        except Exception as e:
            print(f"Error: {e}")
            return "Error:" + str(e)

    def query_voltage_info(self, user_input, graph):
        # 查询知识图谱中与"Battery Voltage"相关的节点
        cypher_query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS 'battery voltage'
        RETURN n
        """
        results = graph.query(cypher_query)
        
        # 将查询结果转换为上下文信息
        context = "\n".join([str(node) for node in results])
        
        # 使用上下文信息生成回答
        if "Battery Voltage is 20V" in user_input:
            question = "What is the possible fault code and solution for Battery Voltage is 20V?"
        elif "System Voltage is 20V" in user_input:
            question = "What is the possible fault code and solution for System Voltage is 20V?"
        else:
            question = "What are the general issues related to Battery Voltage?"
        
        qa_prompt = PromptTemplate(input_variables=["context", "question"], template=self.cypher_qa_prompt)
        response = qa_prompt.format(context=context, question=question)
        
        return response

class update_kg:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
    def process_document(self, file_path, graph):
        file_extension = file_path.split('.')[-1].lower()
        if file_extension == 'docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("不支持的文件格式，目前只支持txt和docx格式")
            
        documents = loader.load()
        
        # 使用更智能的文本分割策略
        text_splitter = CharacterTextSplitter(
            separator="\n\n",  # 按段落分割
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"type of texts: {type(texts)}")
        # 处理每个文本块
        for document in texts:
            print(f'document type: {type(document)}')
            # print(f'document', document)  # 打印前100个字符
            text = document.page_content
            system_prompt = """You are a car maintenance engineer. Please analyze the following text and:
            1. Identify key maintenance concepts, problems, and solutions
            2. For each concept, list common alternative expressions or synonyms
            3. Extract relationships between:
               - Fault codes and their meanings
               - Symptoms and possible causes
               - Components and their functions
               - Problems and solutions
            4. Break down complex information into clear, atomic facts
            5. Ensure consistent terminology while maintaining alternative expressions
            
            Format your response as clear, separate statements that can be easily converted to knowledge graph entries.
            Focus on fault codes, system components, and their relationships.
            """
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text)
            ]
            llm_response = self.llm(messages)
            print("llm_process done")
            print(llm_response.content)
            self.update_graph(llm_response.content, graph)
            
    def update_graph(self, llm_response, graph):
        # 添加同义词处理
        system_prompt = """For the following vehicle diagnostic information, please:
        1. Identify key terms, especially fault codes and components
        2. Generate common synonyms and related terms
        3. Format the response to include both original terms and their variations
        4. Maintain relationships between fault codes, components, and symptoms
        
        Original information:
        """
        print('1')
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=llm_response)
        ]
        enhanced_response = self.llm(messages)
        print('2')
        llm_graph_transformer = LLMGraphTransformer(llm=self.llm)
        print('type of enhanced_response:', type(enhanced_response))
        from langchain.docstore.document import Document
        documents = [Document(page_content=enhanced_response.content)]
        graph_documents = llm_graph_transformer.convert_to_graph_documents(documents)
        print('4')
        # # 添加同义词关系
        # for doc in graph_documents:
        #     if 'relationships' not in doc:
        #         doc['relationships'] = []
        #     # 添加SYNONYM_OF关系
        #     if 'synonyms' in doc:
        #         for term, synonyms in doc['synonyms'].items():
        #             for synonym in synonyms:
        #                 doc['relationships'].append({
        #                     'source': term,
        #                     'target': synonym,
        #                     'type': 'SYNONYM_OF'
        #                 })
        print("5")
        graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
        print("6")