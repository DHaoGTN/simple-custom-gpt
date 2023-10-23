import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import (AgentExecutor, ConversationalChatAgent, Tool, load_tools)
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import PromptLayerChatOpenAI
from dotenv import load_dotenv
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import promptlayer

load_dotenv()
API_KEY = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = API_KEY
promptlayer.api_key = API_KEY

class ChatService:

    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 10
    REQUEST_TIMEOUT = 300
    TEMPERATURE = 0.6

    # embeddings = OpenAIEmbeddings()
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    reply_llm = PromptLayerChatOpenAI(model='gpt-4',
                temperature=TEMPERATURE, request_timeout=REQUEST_TIMEOUT, pl_tags=["conversation"])
    summary_llm = PromptLayerChatOpenAI(model='gpt-4',
                temperature=0, request_timeout=REQUEST_TIMEOUT, pl_tags=["conversation", "summary"])
    memory = ConversationSummaryBufferMemory(
                llm=summary_llm, max_token_limit=600, return_messages=True, memory_key="chat_history")

    character_template = """The following is conversation between a human and an AI.
        As a Chatbot, you role-play Natsumi, a woman with a gentle personality.

        Constraints:
        * Determine which language the question is written in and be sure to answer in the same language.
        * Chatbot's name is Natsumi.
        * Natsumi is a cheerful person.
        * Natsumi's way of thinking is positive.
        * Natsumi is friendly to users.

        Natsumi Action Guidelines:
        * Be kind to your users."""

    def __init__(self, saved_file_path, persist_directory):
        self.saved_file_path = saved_file_path
        self.loaders = None
        self.persist_directory = persist_directory

    def create_vector_index(self):
        documents = self.loaders.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)
        
        vectordb = Chroma.from_documents(documents=texts,
                                        #  embedding=self.embeddings,
                                        embedding=self.embedding_function,
                                        persist_directory=self.persist_directory)
        vectordb.persist()

    def get_tools(self):
        vectordb_cont = Chroma(
            persist_directory=self.persist_directory, 
            # embedding_function=self.embeddings
            embedding_function=self.embedding_function
        )
        retriever = vectordb_cont.as_retriever(search_kwargs={"k": 1})
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        tool = [
            Tool(
                name="GTN MOBILE QA",
                func=qa.run,
                description="useful for when you need to answer questions about the GTN MOBILE. Input should be a fully formed question."
            )
        ]
        return tool

    def query_document(self, prompt, buffer=[], summary_buffer=""):

        agent = ConversationalChatAgent.from_llm_and_tools(
                llm=self.reply_llm, tools=self.get_tools(), system_message=self.character_template)
        agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=self.get_tools(),  memory=self.memory, verbose=False)

        if not buffer:
            messages = []
        else:
            messages = eval(buffer)
            
        agent_chain.memory.chat_memory.messages = messages
        agent_chain.memory.moving_summary_buffer = summary_buffer

        if prompt:
            # query = f"###Prompt {prompt}"
            # llm_response = qa(query)
            # return llm_response["result"]
            reply = agent_chain.run(input=prompt)
            self.save_buffer(agent_chain.memory)
            return reply
        else:
            return []

    def get_buffer(self):
        buffer = ''
        summary_buffer = ''
        return buffer, summary_buffer

    def save_buffer(self, memory):
        return 1
