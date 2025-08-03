# main.py - HackRX API with Pinecone (Fixed Index Issues)
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import aiohttp
import asyncio
import os
import json
import time
import logging
from datetime import datetime
import ssl

# LangChain and AI imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pinecone import Pinecone
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# SSL fix
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize FastAPI
app = FastAPI(
    title="HackRX API with Pinecone",
    description="Production Insurance Policy Analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "hackrx-2024-api-key")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCp26oeIxcV1AcLJ_X8kZEzzP1WCL4ugzc")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_57dCKz_3r3weeg81hm1SSXPTGBvbPstHBzebDy8KCFHEB8Mo1ww1rM7qbsuc8d8AqAwMjw")

# Initialize AI components
try:
    llm = LLM(model="gemini/gemini-1.5-flash-latest")
    genai_model = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    logger.info("✅ AI components initialized")
except Exception as e:
    logger.error(f"❌ AI initialization failed: {str(e)}")
    raise

# Pinecone setup with error handling
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Use an existing index instead of creating new ones
    # First, try to get existing indexes
    existing_indexes = pc.list_indexes()
    
    if existing_indexes:
        # Use the first available index
        index_name = existing_indexes[0].name
        logger.info(f"🔄 Using existing Pinecone index: {index_name}")
        index = pc.Index(index_name)
        
        # Clear the index for fresh data
        try:
            index.delete(delete_all=True)
            logger.info("🧹 Cleared existing index data")
        except Exception as e:
            logger.warning(f"Could not clear index: {str(e)}")
            
    else:
        # If no indexes exist, create one (this might fail on free tier)
        index_name = "hackrx-main"
        try:
            pc.create_index(
                name=index_name,
                dimension=768,
                metric='cosine',
                spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
            )
            time.sleep(10)
            index = pc.Index(index_name)
            logger.info(f"✅ Created new Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"❌ Could not create index: {str(e)}")
            # Fallback to in-memory if Pinecone fails
            index = None
    
    if index:
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        logger.info("✅ Pinecone vector store initialized")
    else:
        vector_store = None
        logger.warning("⚠️ Pinecone not available, will use fallback")
        
except Exception as e:
    logger.error(f"❌ Pinecone initialization failed: {str(e)}")
    index = None
    vector_store = None

# Pydantic models
class HackRXRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# Hybrid processing system (Pinecone + Fallback)
class HybridProcessingSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
        )
        self.processed_docs = []
        self.use_pinecone = vector_store is not None
        
    async def process_document(self, document_url: str):
        """Process document with Pinecone or fallback to in-memory"""
        try:
            # Download document
            os.makedirs("temp", exist_ok=True)
            temp_path = f"temp/document_{int(time.time())}.pdf"
            
            logger.info(f"📥 Downloading: {document_url}")
            
            # Download with SSL bypass
            connector = aiohttp.TCPConnector(ssl=False)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(document_url) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download: HTTP {response.status}"
                        )
                    
                    content = await response.read()
                    with open(temp_path, "wb") as f:
                        f.write(content)
            
            # Process PDF
            loader = PyMuPDFLoader(temp_path)
            documents = loader.load()
            
            # Add metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source_url': document_url,
                    'page_number': i + 1,
                    'total_pages': len(documents),
                    'processed_at': datetime.now().isoformat()
                })
                doc.page_content = self._clean_text(doc.page_content)
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(split_docs):
                chunk.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(split_docs)
                })
            
            # Store documents
            self.processed_docs = split_docs
            
            # Try Pinecone first, fallback to in-memory
            if self.use_pinecone and vector_store:
                try:
                    vector_store.add_documents(split_docs)
                    logger.info(f"✅ Added {len(split_docs)} chunks to Pinecone")
                except Exception as e:
                    logger.warning(f"⚠️ Pinecone storage failed: {str(e)}, using in-memory")
                    self.use_pinecone = False
            else:
                logger.info(f"📚 Stored {len(split_docs)} chunks in memory")
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
            
            return len(split_docs)
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\%\$\d]', '', text)
        return text.strip()
    
    def setup_retriever(self):
        """Setup retriever (Pinecone or BM25 fallback)"""
        try:
            if self.use_pinecone and vector_store:
                # Pinecone semantic retriever
                semantic_retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8}
                )
                
                # BM25 keyword retriever
                if self.processed_docs:
                    bm25_retriever = BM25Retriever.from_documents(
                        self.processed_docs,
                        k=6
                    )
                    
                    # Ensemble retriever
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[semantic_retriever, bm25_retriever],
                        weights=[0.7, 0.3]
                    )
                    return ensemble_retriever
                else:
                    return semantic_retriever
            
            else:
                # Fallback: BM25 only
                if self.processed_docs:
                    return BM25Retriever.from_documents(self.processed_docs, k=10)
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Retriever setup failed: {str(e)}")
            # Final fallback: simple text search
            return None

# Initialize processing system
processing_system = HybridProcessingSystem()

# Search Tools
@tool("HybridDocumentSearch")
def hybrid_document_search(query: str) -> str:
    """Search documents using Pinecone or fallback methods"""
    try:
        retriever = processing_system.setup_retriever()
        
        if retriever:
            docs = retriever.invoke(query)
        else:
            # Manual search fallback
            docs = []
            query_lower = query.lower()
            for doc in processing_system.processed_docs:
                if any(word in doc.page_content.lower() for word in query_lower.split()):
                    docs.append(doc)
            docs = docs[:10]  # Limit results
        
        if not docs:
            return "No relevant information found."
        
        # Format results
        results = []
        for doc in docs:
            page_info = f"[Page {doc.metadata.get('page_number', 'N/A')}]"
            results.append(f"{page_info}\n{doc.page_content}")
        
        return "\n---\n".join(results)
        
    except Exception as e:
        logger.error(f"❌ Search error: {str(e)}")
        return f"Search error: {str(e)}"

@tool("MultiTermSearch")
def multi_term_search(search_terms: List[str]) -> str:
    """Search for multiple terms"""
    try:
        all_results = []
        seen_content = set()
        
        for term in search_terms:
            results = hybrid_document_search(term)
            if results and "No relevant information found" not in results:
                chunks = results.split("\n---\n")
                for chunk in chunks:
                    content_hash = hash(chunk[:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_results.append(chunk)
        
        return "\n---\n".join(all_results[:12])
        
    except Exception as e:
        return f"Multi-term search error: {str(e)}"

# Enhanced QA System
class ProductionQASystem:
    def __init__(self):
        self.setup_agent()
    
    def setup_agent(self):
        self.analyst = Agent(
            role="Expert Insurance Policy Analyst",
            goal="Provide precise, detailed answers about insurance policy terms using advanced document retrieval",
            backstory="""You are a senior insurance analyst with 15+ years experience. You specialize in 
                        extracting exact information from policy documents including waiting periods, grace periods, 
                        coverage amounts, exclusions, and definitions. You always provide specific numbers, 
                        timeframes, and conditions.""",
            tools=[hybrid_document_search, multi_term_search],
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
    
    def answer_question(self, question: str) -> str:
        try:
            task = Task(
                description=f"""Analyze the insurance policy and answer: {question}
                
                Steps:
                1. Use hybrid_document_search for the main search
                2. Use multi_term_search for comprehensive coverage
                3. Extract specific details: numbers, percentages, timeframes, conditions
                4. Provide complete, factual answers
                5. Include relevant policy text when appropriate
                
                Focus on accuracy and specific details from the policy.""",
                expected_output="Precise, factual answer with specific policy details",
                agent=self.analyst
            )
            
            crew = Crew(
                agents=[self.analyst],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            return result.raw.strip()
            
        except Exception as e:
            logger.error(f"❌ QA error: {str(e)}")
            return f"Error processing question: {str(e)}"

# Initialize QA system
qa_system = ProductionQASystem()

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Cleanup function
async def cleanup_resources():
    """Clean up after processing"""
    try:
        if processing_system.use_pinecone and vector_store and index:
            try:
                index.delete(delete_all=True)
                logger.info("🧹 Cleaned up Pinecone index")
            except:
                pass
        
        processing_system.processed_docs = []
        
        if os.path.exists("temp"):
            import shutil
            shutil.rmtree("temp", ignore_errors=True)
            
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "HackRX Production API with Pinecone",
        "pinecone_status": "active" if processing_system.use_pinecone else "fallback",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    try:
        pinecone_status = "unavailable"
        vector_count = 0
        
        if processing_system.use_pinecone and index:
            try:
                stats = index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                pinecone_status = "active"
            except:
                pinecone_status = "error"
        
        return {
            "status": "healthy",
            "pinecone_status": pinecone_status,
            "vector_count": vector_count,
            "documents_in_memory": len(processing_system.processed_docs),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    request: HackRXRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Main HackRX endpoint with Pinecone support"""
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Processing {len(request.questions)} questions")
        logger.info(f"📊 Using: {'Pinecone' if processing_system.use_pinecone else 'In-Memory'}")
        
        # Process document
        chunk_count = await processing_system.process_document(request.documents)
        
        # Process questions
        answers = []
        
        for i, question in enumerate(request.questions, 1):
            logger.info(f"❓ Question {i}/{len(request.questions)}: {question[:50]}...")
            
            try:
                answer = qa_system.answer_question(question)
                answers.append(answer)
                logger.info(f"✅ Answered: {answer[:80]}...")
                
            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                logger.error(f"❌ {error_msg}")
                answers.append(error_msg)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_resources)
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️ Completed in {processing_time:.2f} seconds")
        
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"❌ Request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
