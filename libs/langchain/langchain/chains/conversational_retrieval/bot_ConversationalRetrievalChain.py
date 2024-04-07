# This script, `bot_ConversationalRetrievalChain.py`, is designed to facilitate conversational retrieval using a chain of components including document loading, vector store creation, conversation retrieval from a database, and interaction with a chatbot model. It integrates various LangChain functionalities to process and retrieve relevant information for conversational AI applications.

#ConversationalRetrievalChain
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from dotenv import load_dotenv
import argparse
import os
import json

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv('/home/chatwoot/sheets_chatbot/.env', verbose=True)

import psycopg2
from psycopg2 import sql

verbose_bool = False

dbhost = os.getenv('POSTGRES_HOST')
dbport = '5432'
dbname = 'chatwoot_production'
dbuser = os.getenv('POSTGRES_USERNAME')
dbpass = os.getenv('POSTGRES_PASSWORD')

# Create the database connection
conn = psycopg2.connect(host=dbhost, port=dbport, dbname=dbname, user=dbuser, password=dbpass)

persist_directory = '/home/chatwoot/sheets_chatbot/phpbots-dt/knowledge_base/faiss_vectorstore'
chroma_vectorstore_path = '/home/chatwoot/sheets_chatbot/phpbots-dt/knowledge_base/vectorstore'
chroma_gpt4_vectorstore_path = '/home/chatwoot/sheets_chatbot/phpbots-dt/knowledge_base/vectorstore_gpt4'
documents_path = '/home/chatwoot/sheets_chatbot/phpbots-dt/knowledge_base/documents'

EMBEDDING_MODEL = "text-embedding-3-large"
embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def create_faq_vectorstore():
    csv_file = "faq_dt.csv"
    loader = CSVLoader(file_path=csv_file)
    documents = loader.load()

    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    vectorstore.save_local(persist_directory)

def create_vector_store(documents_path: str, persist_path: str) -> str:
    """
    Create a Vector Store Index from the given documents path and persist it to the specified path.
    
    Args:
        documents_path (str): The path to the directory containing all documents.
        persist_path (str): The path to persist the index.
        
    Returns:
        str: A message indicating the successful creation and persistence of the index.
    """
    # If directory doesnt exist, create it
    if not os.path.exists(persist_path):
        os.makedirs(persist_path)
    
    loader = DirectoryLoader(documents_path, glob="./*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loading {len(documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 0)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=texts, embedding=embedding_function, persist_directory=persist_path)
    vectorstore.persist()
    #vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    #vectorstore.save_local(persist_path)
    
    return f"Vector Store Index created and persisted at {persist_path}"

def get_conversation(conv_id):
    """Retrieve the conversation with the given conv_id."""
    query = sql.SQL("""SELECT conv.id, msg.sender_type, msg.created_at, msg.sender_id, msg.content
                       FROM public.conversations conv
                       RIGHT JOIN public.messages msg on conv.id = msg.conversation_id
                       WHERE conv.id = %s
                       ORDER BY conv.created_at ASC, msg.created_at ASC
                    """)
    with conn.cursor() as cursor:
        cursor.execute(query, (conv_id,))
        messages = cursor.fetchall()
        return [{'conversation_id': message[0],
                 'sender_type': message[1],
                 'created_at': message[2],
                 'sender_id': message[3],
                 'content': message[4]} for message in messages] if messages else []

def format_chat_history(messages):
    if len(messages) > 0:
        msg_array = [HumanMessage(content=msg['content'])
                if msg['sender_type'] == "Contact" 
                else AIMessage(content=msg['content'])
                for msg in messages if msg['content'] and msg['sender_type'] in ["AgentBot", "Contact"]]
        return ChatMessageHistory(messages=msg_array)
    else:
        return ChatMessageHistory(messages=[])

def get_chatbot_chain(vectorstore_path, chat_history):
    llm = ChatOpenAI()
    #vectorstore = FAISS.load_local(persist_directory, OpenAIEmbeddings())
    #retriever = vectorstore.as_retriever()

    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever()
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key='question', output_key='answer')
    memory.chat_memory = chat_history

    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                  retriever=retriever,
                                                  memory=memory,
                                                  verbose=verbose_bool,
                                                  return_source_documents=True
                                                  )
    return chain

def main():
    # Define an argument parser for command line arguments
    parser = argparse.ArgumentParser(description='Process bot.py.')
    parser.add_argument('-c', '--conv_id', type=str, help='Conversation ID')
    parser.add_argument('-m', '--msg', type=str, help='Message')

    # Parse the arguments
    args = parser.parse_args()
    conv_id = args.conv_id
    query = args.msg
    '''print(">>>>>>>>>>>>>>>>>>>>>")
    print("conv_id: ", conv_id)
    print("query: ", query)
    print("<<<<<<<<<<<<<<<<<<<<<")'''

    #create_faq_vectorstore()
    #create_vector_store(documents_path, chroma_gpt4_vectorstore_path)
    
    conv_messages = get_conversation(conv_id)
    chat_history = format_chat_history(conv_messages)
    '''print(">>>>>>>>>>>>>>>>>>>>>")
    print("len conv_messages", len(conv_messages))
    print("conv_messages", conv_messages)
    print("chat_history", chat_history)
    print("<<<<<<<<<<<<<<<<<<<<<")'''

    
    chain = get_chatbot_chain(chroma_gpt4_vectorstore_path, chat_history)
    #query = "Cuéntame como puedo demandar a mi empleador"
    #query = "Que caso puede existir si es que me obliga a trabajar horas extras no pagadas?"
    result = chain.invoke({"question": query})
    links = []
    for doc in result['source_documents']:
        splitter = '# LINK/URL'
        source_array = doc.page_content.split(splitter)
        if len(source_array) > 1:
            links.append(source_array[1])
    
    links_res = '\nLinks útiles:\n' + '\n'.join(links)

    final_result = result['answer'] + links_res if len(links) > 0 else ''
    json_res = {'answer': final_result}
    print(json.dumps(json_res))
    '''print(">>>>>>>>>>>>>>>>>>>>>")
    print("Result: ", result)
    print("Chain: ", chain.memory.chat_memory)
    print("<<<<<<<<<<<<<<<<<<<<<")'''

if __name__ == "__main__":
    main()
