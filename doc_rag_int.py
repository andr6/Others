"""
Enhanced Document RAG System with Advanced Features
"""

import os
import base64
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai
import numpy as np
import pinecone
import torch
import uvicorn
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from langchain_core.documents import Document
from openai import OpenAI
from pydantic import BaseModel, Field
from redis import Redis
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Security Configuration
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize FastAPI
app = FastAPI(
    title="Enhanced Document RAG API",
    description="Retrieval-Augmented Generation system with advanced features",
    version="2.0.0"
)

class SecureConfig:
    """Encrypted configuration and connection management"""
    def __init__(self):
        self.fernet = Fernet(os.environ["ENCRYPTION_KEY"])
        self.redis = Redis.from_url(os.environ["REDIS_URL"])
        self.cache_ttl = int(os.environ.get("CACHE_TTL", 3600))
        
    def get_secret(self, key: str) -> str:
        encrypted = os.environ.get(key)
        if not encrypted:
            raise ValueError(f"Missing {key} in environment")
        return self.fernet.decrypt(encrypted.encode()).decode()

config = SecureConfig()

# Pydantic Models
class DocumentRAGState(BaseModel):
    """Enhanced state model with version tracking and metrics"""
    question: str
    document_path: str
    pages_as_base64_jpeg_images: List[str] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    relevant_documents: List[Document] = Field(default_factory=list)
    response: Optional[str] = None
    version: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    namespace: str = "default"
    confidence: Optional[float] = None
    metrics: Dict = Field(default_factory=dict)
    content_hash: Optional[str] = None

# Embedding Classes
class GPUEmbeddings:
    """GPU-accelerated local embeddings with fallback"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query], convert_to_numpy=True)[0].tolist()

class OpenAIEmbeddings:
    """OpenAI embeddings with rate limiting"""
    def __init__(self):
        self.client = OpenAI(api_key=config.get_secret("OPENAI_API_KEY"))
        
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [embedding.embedding for embedding in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]

# Core RAG Agent
class DocumentRAGAgent:
    """Enhanced RAG agent with hybrid search and advanced features"""
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash-002",
        k: int = 5,
        storage_backend: str = "pinecone",
        embedding_type: str = "chroma",
        namespace: str = "default",
        hybrid_ratio: float = 0.7
    ):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.k = k
        self.embedding_type = embedding_type
        self.namespace = namespace
        self.hybrid_ratio = hybrid_ratio
        self.vector_store = self._init_vector_store(storage_backend)
        self.embedder = self._init_embedder()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.batch_size = 100

    def _init_embedder(self):
        embedders = {
            "chroma": embedding_functions.DefaultEmbeddingFunction(),
            "openai": OpenAIEmbeddings(),
            "local": GPUEmbeddings()
        }
        return embedders.get(self.embedding_type, embedders["chroma"])

    def _init_vector_store(self, backend: str):
        if backend != "pinecone":
            raise ValueError("Currently only Pinecone is supported")

        pinecone.init(
            api_key=config.get_secret("PINECONE_API_KEY"),
            environment=os.environ.get("PINECONE_ENV", "production")
        )
        
        index_name = f"{os.environ.get('ENV_PREFIX', 'dev')}-document-rag"
        dimension = 1536 if self.embedding_type == "openai" else 384
        
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                pods=1,
                replicas=1,
                pod_type="p1.x2"
            )
            
        return pinecone.Index(index_name)

    async def index_documents(self, state: DocumentRAGState) -> Dict:
        """Enhanced indexing with version control and bulk processing"""
        if not state.documents:
            raise ValueError("No documents to index")

        try:
            # Parallel embedding generation
            vectors = await self._generate_vectors(state.documents)
            
            # Batch processing with connection pooling
            results = []
            for batch in self._chunk_list(vectors, self.batch_size):
                result = await run_in_threadpool(
                    self.vector_store.upsert,
                    vectors=batch,
                    namespace=self.namespace
                )
                results.append(result)
                
            state.metrics.update({
                "indexed_vectors": len(vectors),
                "batches_processed": len(results)
            })
            return {"status": "success", "metrics": state.metrics}
            
        except Exception as e:
            logger.error(f"Indexing failed: {str(e)}")
            raise

    async def _generate_vectors(self, documents: List[Document]) -> List[Tuple]:
        """Generate vectors with content hashing for version control"""
        futures = []
        for doc in documents:
            futures.append(
                self.executor.submit(
                    self._process_document,
                    doc
                )
            )
        return [future.result() for future in futures]

    def _process_document(self, doc: Document) -> Tuple:
        content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        return (
            f"doc-{content_hash[:12]}",
            self.embedder.embed_documents([doc.page_content])[0],
            {
                "content_hash": content_hash,
                "truncated_content": doc.page_content[:995] + "[...]" if len(doc.page_content) > 1000 else doc.page_content,
                "version": datetime.now().isoformat(),
                **doc.metadata
            }
        )

    async def query(self, state: DocumentRAGState) -> Dict:
        """Hybrid search with caching and metrics"""
        cache_key = f"{self.namespace}:{hashlib.sha256(state.question.encode()).hexdigest()}"
        if cached := config.redis.get(cache_key):
            return json.loads(cached)

        start_time = time.time()
        vector_results, keyword_results = await self._parallel_search(state.question)
        combined_results = self._combine_results(vector_results, keyword_results)
        
        response = await self._generate_response(
            state.question,
            combined_results,
            start_time
        )
        
        config.redis.setex(cache_key, config.cache_ttl, json.dumps(response))
        return response

    async def _parallel_search(self, query: str) -> Tuple:
        """Parallel vector and keyword search"""
        vector_future = run_in_threadpool(
            self.vector_store.query,
            vector=self.embedder.embed_query(query),
            top_k=self.k,
            include_metadata=True,
            namespace=self.namespace
        )
        
        keyword_future = run_in_threadpool(
            self._keyword_search,
            query
        )
        
        return await vector_future, await keyword_future

    def _keyword_search(self, query: str) -> Dict:
        """Simplified keyword search (BM25 placeholder)"""
        return {"matches": []}  # Implement actual keyword search here

    def _combine_results(self, vector_res: Dict, keyword_res: Dict) -> List[Document]:
        """Combine results using hybrid ratio scoring"""
        scored = []
        for match in vector_res.get("matches", []):
            scored.append((
                match["metadata"],
                match["score"] * self.hybrid_ratio
            ))
            
        for match in keyword_res.get("matches", []):
            scored.append((
                match["metadata"],
                match["score"] * (1 - self.hybrid_ratio)
            ))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            Document(
                page_content=item[0]["truncated_content"],
                metadata=item[0]
            ) for item in scored[:self.k]
        ]

    async def _generate_response(self, question: str, docs: List[Document], start_time: float) -> Dict:
        """Generate response with confidence scoring and source highlighting"""
        context = "\n".join([
            f"[[Document {i+1}]] (Page {doc.metadata.get('page_number', '?')}): "
            f"{self._highlight_relevant(doc.page_content, question)}"
            for i, doc in enumerate(docs)
        ])
        
        prompt = f"""Answer the question based on the context below. 
        Include page references and a confidence estimate.
        If uncertain, state that clearly.

        Question: {question}
        Context: {context}
        """
        
        response = await run_in_threadpool(
            self.model.generate_content,
            prompt
        )
        
        return {
            "response": response.text,
            "confidence": self._calculate_confidence(docs),
            "sources": [doc.metadata for doc in docs],
            "metrics": {
                "retrieval_time": time.time() - start_time,
                "context_length": len(context),
                "sources_used": len(docs)
            }
        }

    def _calculate_confidence(self, docs: List[Document]) -> float:
        """Calculate confidence based on similarity scores"""
        scores = [doc.metadata.get("score", 0) for doc in docs]
        return min(np.mean(scores) * 100, 100) if scores else 0

    def _highlight_relevant(self, text: str, query: str) -> str:
        """Simple text highlighting for relevant terms"""
        for term in query.split():
            if term.lower() in text.lower():
                text = text.replace(term, f"**{term}**")
        return text

    @staticmethod
    def _chunk_list(lst: List, n: int):
        """Yield successive n-sized chunks from list"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

# API Endpoints
@app.post("/v1/index")
async def index_endpoint(
    path: str,
    namespace: str = "default",
    api_key: str = Security(api_key_header)
):
    """Index documents endpoint"""
    if not validate_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        agent = DocumentRAGAgent(namespace=namespace)
        state = DocumentRAGState(document_path=path, namespace=namespace)
        return await agent.index_documents(state)
    except Exception as e:
        log_audit_event("index_failed", path, error=str(e))
        raise HTTPException(500, str(e))

@app.post("/v1/query")
async def query_endpoint(
    question: str,
    namespace: str = "default",
    api_key: str = Security(api_key_header)
):
    """Query endpoint with rate limiting"""
    if not check_rate_limit(api_key):
        raise HTTPException(429, "Rate limit exceeded")
    
    try:
        agent = DocumentRAGAgent(namespace=namespace)
        state = DocumentRAGState(question=question, namespace=namespace)
        return await agent.query(state)
    except Exception as e:
        log_audit_event("query_failed", question, error=str(e))
        raise HTTPException(500, str(e))

# Security Functions
def validate_api_key(key: str) -> bool:
    return config.redis.exists(f"api_key:{key}") == 1

def check_rate_limit(key: str) -> bool:
    current_window = int(time.time()) // 60
    count = config.redis.incr(f"ratelimit:{key}:{current_window}")
    config.redis.expire(f"ratelimit:{key}:{current_window}", 60)
    return count <= 100

def log_audit_event(action: str, target: str, **kwargs):
    event = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "target": target,
        **kwargs
    }
    config.redis.lpush("audit_log", json.dumps(event))

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Document RAG System")
    parser.add_argument("--path", type=str, help="Document path")
    parser.add_argument("--question", type=str, help="Query question")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    args = parser.parse_args()

    if args.serve:
        print(f"Starting API server with GPU: {torch.cuda.is_available()}")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Command-line mode
        if args.path:
            agent = DocumentRAGAgent()
            state = DocumentRAGState(document_path=args.path)
            agent.index_documents(state)
        elif args.question:
            agent = DocumentRAGAgent()
            state = DocumentRAGState(question=args.question)
            print(agent.query(state))
