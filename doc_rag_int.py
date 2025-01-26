import os
import base64
from pathlib import Path
from typing import Optional
import argparse
import time

import google.generativeai as genai
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from document_ai_agents.logger import logger

class ChromaEmbeddingsAdapter:
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):
        return self.ef(texts)

    def embed_query(self, query):
        return self.ef([query])[0]

class OpenAIEmbeddings:
    def __init__(self):
        self.client = OpenAI()

    def embed_documents(self, texts):
        response = self.client.embeddings.create(input=texts, model="text-embedding-ada-002")
        return [embedding.embedding for embedding in response.data]

    def embed_query(self, query):
        response = self.client.embeddings.create(input=[query], model="text-embedding-ada-002")
        return response.data[0].embedding

class DocumentRAGState(BaseModel):
    question: str
    document_path: str
    pages_as_base64_jpeg_images: list[str] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    relevant_documents: list[Document] = Field(default_factory=list)
    response: Optional[str] = None

class DocumentRAGAgent:
    def __init__(self, model_name="gemini-1.5-flash-002", k=3, storage_backend="pinecone", embedding_type="chroma"):
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)
        self.k = k
        self.embedding_type = embedding_type
        self.vector_store = self._initialize_vector_store(storage_backend)
        self.embedder = self._initialize_embedder()

    def _initialize_embedder(self):
        if self.embedding_type == "chroma":
            return ChromaEmbeddingsAdapter(embedding_functions.DefaultEmbeddingFunction())
        elif self.embedding_type == "openai":
            return OpenAIEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")

    def _initialize_vector_store(self, backend):
        if backend == "pinecone":
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
            pc = Pinecone(api_key=pinecone_api_key)
            index_name = "document-rag"

            dimension = 1536 if self.embedding_type == "openai" else 384

            if index_name in pc.list_indexes().names():
                index = pc.Index(index_name)
                index_stats = index.describe_index_stats()
                existing_dimension = index_stats.dimension

                if existing_dimension != dimension:
                    logger.warning(f"Existing index dimension ({existing_dimension}) does not match required dimension ({dimension}).")
                    logger.warning("Deleting existing index and creating a new one with the correct dimension.")
                    pc.delete_index(index_name)
                    pc.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region=pinecone_region)
                    )
            else:
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=pinecone_region)
                )

            return pc.Index(index_name)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def index_documents(self, state: DocumentRAGState):
        if not state.documents:
            raise ValueError("No documents found to index. Ensure parsing is successful.")

        logger.info(f"Indexing {len(state.documents)} documents in Pinecone.")

        vectors = []
        for i, doc in enumerate(state.documents):
            embedding = self.embedder.embed_documents([doc.page_content])[0]
            # Flatten and stringify the metadata
            metadata = {
                "document_path": str(doc.metadata.get("document_path", "")),
                "page_number": str(doc.metadata.get("page_number", "")),
                "content": doc.page_content[:1000]  # Truncate content if necessary
            }
            vectors.append((f"doc-{i}", embedding.tolist(), metadata))

        try:
            self.vector_store.upsert(vectors)
            logger.info(f"Successfully indexed {len(vectors)} vectors in Pinecone.")
        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            raise

    def answer_question(self, state: DocumentRAGState):
        if not state.documents:
            raise ValueError("No documents available to answer the question.")

        logger.info(f"Searching for relevant documents for the query: {state.question}")

        query_embedding = self.embedder.embed_query(state.question)

        query_results = self.vector_store.query(
            vector=query_embedding.tolist(),
            top_k=self.k,
            include_metadata=True,
        )

        if not query_results.get("matches"):
            return {"response": "No relevant documents found.", "relevant_documents": []}

        relevant_docs = [
            Document(
                page_content=match["metadata"]["content"],
                metadata={k: v for k, v in match["metadata"].items() if k != "content"}
            )
            for match in query_results["matches"]
        ]

        logger.info(f"Retrieved {len(relevant_docs)} relevant documents.")

        # Generate a response using the relevant documents
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Based on the following context, answer the question: '{state.question}'\n\nContext:\n{context}\n\nAnswer:"

        response = self.model.generate_content(prompt)
        answer = response.text

        return {"response": answer, "relevant_documents": relevant_docs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents and interact with their content.")
    parser.add_argument("--path", type=str, required=True, help="Path to the document or directory of documents.")
    parser.add_argument("--storage", type=str, default="pinecone", help="Vector store backend: 'pinecone' or 'chroma'.")
    parser.add_argument("--embedding", type=str, default="chroma", choices=["chroma", "openai"], help="Embedding type to use: 'chroma' or 'openai'.")
    args = parser.parse_args()

    document_path = args.path
    storage_backend = args.storage
    embedding_type = args.embedding

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
    agent2 = DocumentRAGAgent(storage_backend=storage_backend, embedding_type=embedding_type)

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

        agent2.index_documents(state2)
        print(f"Indexed: {document_file}")

    print("Interactive Question-Answering Session. Type 'exit' to quit.")
    while True:
        question = input("Enter your question (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break

        for document_file in document_files:
            state2.question = question  # Reuse state with the new question

            try:
                result = agent2.answer_question(state2)
                print(f"Answer from {document_file}: {result['response']}")
            except Exception as e:
                logger.error(f"Error while answering question: {e}")
