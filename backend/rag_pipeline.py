
#os.environ["OPENAI_API_KEY"] = "sk-8tyGI_x9CqUw_YyT8MRKXocRXX0nixlm6YtwOT9JzdT3BlbkFJ8cYJHuSIdfWYyZwjdo8O7-iV_MBwORz4ctUHfd5RwA"
#os.environ.get("OPENAI_API_KEY")

# rag_pipeline.py
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
import pandas as pd
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import HTTPException



llm = ChatOpenAI(model="gpt-3.5-turbo")

vector_store = None
rag_chain = None

def load_and_process_data():
    try:
        # Load Excel files
        file_path_1 = 'backend/data/Christmas Research Results.xlsx'
        file_path_2 = 'backend/data/Sustainability Research Results.xlsx'
        df1 = pd.read_excel(file_path_1)
        df2 = pd.read_excel(file_path_2)

        # Add dataset labels for comparison purposes
        df1["source"] = "Christmas"
        df2["source"] = "Sustainability"
        
        combined_data = pd.concat([df1, df2])

        # Save the combined data as CSV
        combined_csv_path = 'combined_data.csv'
        combined_data.to_csv(combined_csv_path, index=False)

        # Process the data using CSVLoader
        loader = CSVLoader(file_path=combined_csv_path)
        docs = loader.load_and_split()

        # Tag documents with their dataset source for comparison purposes
        for doc in docs:
            if "Christmas" in doc.page_content:
                doc.metadata["source"] = "Christmas"
            elif "Sustainability" in doc.page_content:
                doc.metadata["source"] = "Sustainability"

        return docs
    except Exception as e:
        raise Exception(f"Error loading Excel files: {str(e)}")

# Initialize RAG pipeline (called once during server startup)
def initialize_rag_pipeline_once():
    global vector_store, rag_chain
    try:
        documents = load_and_process_data()

        # Initialize FAISS vector store and OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))  # Initialize FAISS index
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # Add the split CSV data to the vector store
        vector_store.add_documents(documents=documents)

        # Create the retriever
        retriever = vector_store.as_retriever()

        # Set up the system prompt for comparison
        system_prompt = (
            "You are an assistant for analyzing and comparing survey data from two different datasets. "
            "You have information from Christmas and Sustainability surveys. "
            "Use the following retrieved context from both datasets to answer the question. "
            "Highlight key similarities and differences if applicable. "
            "If the answer cannot be found in both datasets, provide insights from one of them. "
            "\n\n"
            "{context}"
        )

        # Create the prompt template for the chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Create the question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        print("RAG pipeline initialized successfully!")
    except Exception as e:
        raise Exception(f"Error initializing RAG pipeline: {str(e)}")

# Query the RAG system (this is called for each query)
def query_rag_system(query: str):
    try:
        # Ensure the RAG pipeline is initialized before querying
        if not rag_chain:
            raise Exception("RAG pipeline is not initialized.")
        
        # Query the RAG system with the user's question
        answer = rag_chain.invoke({"input": query})
        return answer['answer']
    except Exception as e:
        raise Exception(f"Error querying RAG system: {str(e)}")
