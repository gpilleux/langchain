import argparse
import json
import os

import psycopg2
from dotenv import load_dotenv
from langchain.chains.conversational_retrieval import \
    ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llm import ChatOpenAI
from langchain_community.chat_message_histories.in_memory import \
    ChatMessageHistory
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import VectorStore

load_dotenv(verbose=True)

dbhost = os.getenv('POSTGRES_HOST')
dbport = '5432'
dbname = 'chatwoot_production'
dbuser = os.getenv('POSTGRES_USERNAME')
dbpass = os.getenv('POSTGRES_PASSWORD')

def create_database_connection():
    return psycopg2.connect(host=dbhost, port=dbport, dbname=dbname, user=dbuser, password=dbpass)

def create_vector_store(documents_path, persist_path):
    loader = CSVLoader(file_path=documents_path)
    documents = loader.load()
    vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
    vectorstore.save_local(persist_path)
    return vectorstore

def format_chat_history(messages):
    msg_array = [HumanMessage(content=msg['content']) if msg['sender_type'] == "Contact" else AIMessage(content=msg['content']) for msg in messages if msg['content'] and msg['sender_type'] in ["AgentBot", "Contact"]]
    return ChatMessageHistory(messages=msg_array)

def get_conversation(conv_id, conn):
    with conn.cursor() as cursor:
        cursor.execute("""SELECT conv.id, msg.sender_type, msg.created_at, msg.sender_id, msg.content FROM public.conversations conv RIGHT JOIN public.messages msg on conv.id = msg.conversation_id WHERE conv.id = %s ORDER BY conv.created_at ASC, msg.created_at ASC""", (conv_id,))
        messages = cursor.fetchall()
        return [{'conversation_id': message[0], 'sender_type': message[1], 'created_at': message[2], 'sender_id': message[3], 'content': message[4]} for message in messages] if messages else []

def main():
    parser = argparse.ArgumentParser(description='Process bot.py.')
    parser.add_argument('-c', '--conv_id', type=str, help='Conversation ID')
    parser.add_argument('-m', '--msg', type=str, help='Message')
    args = parser.parse_args()

    conn = create_database_connection()
    conv_messages = get_conversation(args.conv_id, conn)
    chat_history = format_chat_history(conv_messages)

    llm = ChatOpenAI()
    vectorstore_path = '/path/to/vectorstore'
    vectorstore = VectorStore.load_local(vectorstore_path)
    retriever = vectorstore.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, verbose=True)
    result = chain.invoke({"question": args.msg})
    print(json.dumps({'answer': result['answer']}))

if __name__ == "__main__":
    main()
