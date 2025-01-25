import os
import base64
from pathlib import Path
from typing import Optional
import argparse

import google.generativeai as genai
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from document_ai_agents.logger import logger

class ChromaEmbeddingsAdapter(Embeddings):
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]

class DocumentRAGState(BaseModel):
    question: str
    document_path: str
    pages_as_base64_jpeg_images: list[str] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    relevant_documents: list[Document] = Field(default_factory=list)
    response: Optional[str] = None

class DocumentRAGAgent:
    def __init__(self, model_name="gemini-1.5-flash-002", k=3, storage_backend="chroma"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
        )
        self.vector_store = self._initialize_vector_store(storage_backend)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        self.graph = None
        self.build_agent()

    def _initialize_vector_store(self, backend):
        if backend == "pinecone":
            from langchain.vectorstores import Pinecone
            import pinecone
            pinecone.init(api_key="your-pinecone-api-key")
            return Pinecone(index_name="document-rag")
        elif backend == "weaviate":
            from langchain.vectorstores import Weaviate
            return Weaviate(client_config={"host": "your-weaviate-instance-url"})
        else:
            return Chroma(
                collection_name="document-rag",
                embedding_function=ChromaEmbeddingsAdapter(
                    embedding_functions.DefaultEmbeddingFunction()
                ),
            )

    def index_documents(self, state: DocumentRAGState):
        if not state.documents:
            logger.error("No documents found to index. Please ensure the parsing step succeeded and check the input document format.")
            print("Error: No documents to index. Ensure your document parsing is successful.")
            raise ValueError("Documents should have at least one element.")

        existing_ids = self.vector_store.get(where={"document_path": state.document_path}).get("ids", [])
        if existing_ids:
            logger.info(f"Documents for {state.document_path} are already indexed.")
            return

        logger.info(f"Indexing {len(state.documents)} documents for {state.document_path}.")
        for doc in state.documents:
            if not isinstance(doc, Document):
                logger.error("Invalid document format. Each document should be an instance of the Document class.")
                raise ValueError("Document is not properly formatted.")

        try:
            self.vector_store.add_documents(state.documents)
            logger.info("Documents successfully indexed.")
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise

    def answer_question(self, state: DocumentRAGState):
        if not state.documents:
            logger.error(f"No documents available for answering the question: {state.question}")
            raise ValueError("Documents should have at least one element.")

        relevant_documents: list[Document] = self.retriever.invoke(state.question)

        if not relevant_documents:
            logger.warning(f"No relevant documents retrieved for question: {state.question}. Consider refining your question or ensuring the documents are properly indexed.")
            return {"response": "No relevant documents found. Please refine your question or check the indexed data.", "relevant_documents": []}

        images = list(
            set(
                [
                    state.pages_as_base64_jpeg_images[doc.metadata.get("page_number", 0)]
                    for doc in relevant_documents
                    if "page_number" in doc.metadata
                ]
            )
        )  # Avoid duplicates

        logger.info(f"Responding to question: {state.question}")
        messages = (
            [{"mime_type": "image/jpeg", "data": base64_jpeg} for base64_jpeg in images]
            + [doc.page_content for doc in relevant_documents]
            + [
                f"Answer this question using the context images and text elements only: {state.question}",
            ]
        )

        try:
            response = self.model.generate_content(messages)
        except TimeoutError:
            logger.error("Model request timed out. Returning fallback response.")
            response = type('FallbackResponse', (object,), {"text": "The request timed out. Please try again later."})()

        return {"response": response.text, "relevant_documents": relevant_documents}

    def build_agent(self):
        builder = StateGraph(DocumentRAGState)
        builder.add_node("index_documents", self.index_documents)
        builder.add_node("answer_question", self.answer_question)

        builder.add_edge(START, "index_documents")
        builder.add_edge("index_documents", "answer_question")
        builder.add_edge("answer_question", END)
        self.graph = builder.compile()

# Function to display the help menu

def run_tests():
    print("Running Tests...\n")

    # Test 1: Ensure parsing extracts documents
    try:
        print("Test 1: Parsing documents")
        test_state = DocumentLayoutParsingState(document_path="../document_ai_agents/data/docs.pdf")
        parsing_result = agent1.graph.invoke(test_state)
        assert parsing_result["documents"], "Parsing failed. No documents extracted."
        print("Test 1 Passed: Documents parsed successfully.\n")
    except AssertionError as e:
        print(f"Test 1 Failed: {e}\n")

    # Test 2: Ensure indexing works correctly
    try:
        print("Test 2: Indexing documents")
        state_to_index = DocumentRAGState(
            question="",
            document_path="../document_ai_agents/data/docs.pdf",
            pages_as_base64_jpeg_images=parsing_result["pages_as_base64_jpeg_images"],
            documents=parsing_result["documents"],
        )
        agent2.index_documents(state_to_index)
        print("Test 2 Passed: Documents indexed successfully.\n")
    except Exception as e:
        print(f"Test 2 Failed: {e}\n")

    # Test 3: Interaction with the indexed data
    try:
        print("Test 3: Interaction with indexed data")
        state_to_query = DocumentRAGState(
            question="Who is the author?",
            document_path="../document_ai_agents/data/docs.pdf",
        )
        query_result = agent2.answer_question(state_to_query)
        assert query_result["response"], "Query failed. No response returned."
        print(f"Test 3 Passed: Query successful. Response: {query_result['response']}\n")
    except AssertionError as e:
        print(f"Test 3 Failed: {e}\n")

    print("All Tests Completed.")

def display_help():
    print("""
    Usage:
    python script.py --path <PDF_OR_DIRECTORY_PATH>

    Arguments:
    --path : Path to a single PDF file or a directory containing multiple PDFs.

    Commands:
    After processing, you can ask questions about the content interactively.
    Type 'exit' to quit the interactive session.
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents and interact with their content.")
    parser.add_argument("--path", type=str, required=True, help="Path to the document or directory of documents.")
    parser.add_argument("--storage", type=str, default="chroma", help="Vector store backend: 'chroma', 'pinecone', or 'weaviate'.")
    args = parser.parse_args()

    document_path = args.path
    storage_backend = args.storage

    from document_ai_agents.document_parsing_agent import (
        DocumentLayoutParsingState,
        DocumentParsingAgent,
    )

    def process_documents(document_path):
        document_files = []
        if os.path.isdir(document_path):
            document_files = [
                str(Path(document_path) / file)
                for file in os.listdir(document_path)
                if file.endswith((".pdf", ".docx", ".pptx"))
            ]
        elif document_path.endswith((".pdf", ".docx", ".pptx")):
            document_files = [document_path]
        else:
            raise ValueError("Invalid file or directory. Provide a supported document or a directory containing documents.")
        return document_files

    document_files = process_documents(document_path)

    agent1 = DocumentParsingAgent()
    agent2 = DocumentRAGAgent(storage_backend=storage_backend)

    for document_file in document_files:
        print(f"Processing: {document_file}")

        state1 = DocumentLayoutParsingState(document_path=document_file)
        result1 = agent1.graph.invoke(state1)

        if not result1["documents"]:
            logger.error(f"Parsing failed for {document_file}. No documents extracted.")
            continue

        logger.info(f"Parsed {len(result1['documents'])} documents from {document_file}.")

        state2 = DocumentRAGState(
            question="",
            document_path=document_file,
            pages_as_base64_jpeg_images=result1["pages_as_base64_jpeg_images"],
            documents=result1["documents"],
        )

        agent2.graph.invoke(state2)
        print(f"Indexed: {document_file}")

    print("\nInteractive Question-Answering Session. Type 'exit' to quit.")
    while True:
        question = input("Enter your question (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break

        for document_file in document_files:
            state2.question = question  # Reuse state with the new question

            try:
                result = agent2.graph.invoke(state2)
                print(f"Answer from {document_file}: {result['response']}")
            except Exception as e:
                logger.error(f"Error while answering question: {e}")
