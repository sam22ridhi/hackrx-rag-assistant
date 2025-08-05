# main.py - Optimized HackRX API with Enhanced RAG
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import certifi
import aiohttp
import os
import json
import time
import logging
from datetime import datetime

# LangChain and AI imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from pinecone import Pinecone
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="HackRX Enhanced RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Security
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "hackrx-2024-api-key")

# Initialize AI components
llm = LLM(model="gemini/gemini-1.5-flash-latest")
genai_model = GoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key="AIzaSyCb_qbNOJSjUuab72-7sk-otRJelTSFSaU",
    temperature=0.1
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key="AIzaSyCb_qbNOJSjUuab72-7sk-otRJelTSFSaU"
)

# Pinecone setup
pc = Pinecone(api_key="pcsk_57dCKz_3r3weeg81hm1SSXPTGBvbPstHBzebDy8KCFHEB8Mo1ww1rM7qbsuc8d8AqAwMjw")
index_name = "hackrx-enhanced-rag"

try:
    index = pc.Index(index_name)
except:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
    )
    time.sleep(10)
    index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Pydantic models
class HackRXRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# Enhanced RAG System
class EnhancedRAGSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.processed_docs = []
        
    async def process_document(self, document_url: str):
        try:
            os.makedirs("temp", exist_ok=True)
            temp_path = f"temp/document_{int(time.time())}.pdf"

            async with aiohttp.ClientSession() as session:
                async with session.get(document_url) as response:
                    if response.status != 200:
                        print("Failed to fetch document. Status:", response.status)
                        raise HTTPException(status_code=400, detail="Failed to download document")

                    content = await response.read()
                    with open(temp_path, "wb") as f:
                        f.write(content)

            await asyncio.sleep(0.5)  # Make sure file is flushed

        # Load and process
            # Process with enhanced metadata
            loader = PyMuPDFLoader(temp_path)
            documents = loader.load()
            
            # Add rich metadata and clean text
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source_url': document_url,
                    'page': i + 1,
                    'file_name': f"policy_document.pdf",
                    'processed_at': datetime.now().isoformat(),
                    'doc_type': 'policy'
                })
                # Clean text
                doc.page_content = self._clean_text(doc.page_content)
            
            # Enhanced chunking with overlap
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(split_docs):
                chunk.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(split_docs)
                })
            
            # Store for BM25 retriever
            self.processed_docs = split_docs
            
            # Add to vector store
            vector_store.add_documents(split_docs)
            
            # Cleanup
            os.remove(temp_path)
            
            return len(split_docs)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        import re
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\%\$]', '', text)
        return text.strip()
    
    def setup_hybrid_retriever(self):
        """Setup hybrid retrieval combining semantic and keyword search"""
        try:
            # Semantic retriever (existing)
            semantic_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )
            
            # BM25 keyword retriever
            bm25_retriever = BM25Retriever.from_documents(
                self.processed_docs,
                k=6
            )
            
            # Ensemble retriever combining both
            ensemble_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.6, 0.4]  # Favor semantic slightly
            )
            
            return ensemble_retriever
            
        except Exception as e:
            # Fallback to semantic only
            return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize RAG system
rag_system = EnhancedRAGSystem()

# Enhanced Search Tools
@tool("HybridDocumentSearch")
def hybrid_document_search(query: str) -> str:
    """
    Advanced hybrid search combining semantic similarity and keyword matching
    with relevance scoring and deduplication.
    """
    try:
        retriever = rag_system.setup_hybrid_retriever()
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant information found."
        
        # Remove duplicates and format
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Format with metadata
        results = []
        for doc in unique_docs[:10]:  # Top 10 most relevant
            source_info = f"[Page {doc.metadata.get('page', 'N/A')}]"
            results.append(f"{source_info}\n{doc.page_content}")
        
        return "\n---\n".join(results)
        
    except Exception as e:
        return f"Search error: {str(e)}"

@tool("MultiQuerySearch")
def multi_query_search(queries: List[str]) -> str:
    """
    Execute multiple strategic searches and consolidate unique results
    with relevance ranking.
    """
    try:
        all_results = []
        seen_content = set()
        
        for query in queries:
            results = hybrid_document_search(query)
            if results and "No relevant information found" not in results:
                # Split by separator and process each chunk
                chunks = results.split("\n---\n")
                for chunk in chunks:
                    content_hash = hash(chunk[:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_results.append(chunk)
        
        return "\n---\n".join(all_results[:15])  # Top 15 most relevant
        
    except Exception as e:
        return f"Multi-query search error: {str(e)}"

@tool("AnswerValidator")
def answer_validator(question: str, proposed_answer: str, evidence: str) -> str:
    """
    Validate the proposed answer against the evidence and question
    to ensure accuracy and completeness.
    """
    try:
        validation_prompt = f"""
        Question: {question}
        Proposed Answer: {proposed_answer}
        Evidence: {evidence}
        
        Validate if the proposed answer:
        1. Directly answers the question
        2. Is supported by the evidence
        3. Includes all relevant details (numbers, conditions, etc.)
        4. Is factually accurate
        
        Return: VALID if good, or INVALID with corrections needed.
        """
        
        # Simple validation logic
        if len(proposed_answer.strip()) < 10:
            return "INVALID - Answer too brief"
        
        if "I don't know" in proposed_answer or "not found" in proposed_answer.lower():
            return "INVALID - Non-informative answer"
        
        # Check if answer contains specific details
        has_specifics = any(char.isdigit() for char in proposed_answer) or \
                       any(word in proposed_answer.lower() for word in ['days', 'months', 'years', 'percent', '%', 'rupees', 'amount'])
        
        if not has_specifics and "what is" in question.lower():
            return "INVALID - Missing specific details"
        
        return "VALID"
        
    except Exception as e:
        return f"Validation error: {str(e)}"

# Enhanced Question Answering Crew
class EnhancedQACrew:
    def __init__(self):
        self.setup_agents()
    
    def setup_agents(self):
        # 1. Query Analyzer Agent
        self.query_analyzer = Agent(
            role="Expert Query Analyst",
            goal="Analyze questions to identify key information needed and generate optimal search strategies",
            backstory="""You are a master at understanding insurance and policy questions. You can break down 
                        complex questions into searchable components and identify what specific information 
                        is needed to provide complete answers.""",
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
        
        # 2. Research Agent with Enhanced Tools
        self.researcher = Agent(
            role="Strategic Policy Researcher",
            goal="Execute comprehensive searches to gather all relevant information for answering questions",
            backstory="""You are an expert researcher who uses multiple search strategies to find comprehensive 
                        information. You know how to search for synonyms, related terms, and cross-reference 
                        information to ensure nothing is missed.""",
            tools=[hybrid_document_search, multi_query_search],
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
        
        # 3. Answer Synthesizer Agent
        self.synthesizer = Agent(
            role="Expert Answer Synthesizer",
            goal="Create clear, accurate, and complete answers based on research findings",
            backstory="""You are skilled at taking complex policy information and converting it into clear, 
                        direct answers. You always include specific details like numbers, timeframes, and 
                        conditions when available.""",
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
        
        # 4. Quality Validator Agent
        self.validator = Agent(
            role="Answer Quality Validator",
            goal="Ensure answers are accurate, complete, and directly address the questions asked",
            backstory="""You are a quality assurance expert who validates that answers meet high standards 
                        of accuracy and completeness. You catch missing details and ensure answers are 
                        properly formatted.""",
            tools=[answer_validator],
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
    
    def answer_question(self, question: str) -> str:
        """Process a single question with enhanced multi-agent approach"""
        try:
            # Task 1: Analyze query
            analysis_task = Task(
                description=f"""Analyze this question to identify:
                1. Key terms and concepts to search for
                2. Type of information needed (timeframes, amounts, conditions, etc.)
                3. Potential synonyms and related terms
                Question: {question}""",
                expected_output="Analysis of the question with search strategy recommendations",
                agent=self.query_analyzer
            )
            
            # Task 2: Research information
            research_task = Task(
                description=f"""Based on the query analysis, execute strategic searches to find all relevant 
                information for: {question}
                Use both single searches and multi-query approaches to ensure comprehensive coverage.""",
                expected_output="Comprehensive research findings with all relevant policy information",
                agent=self.researcher,
                context=[analysis_task]
            )
            
            # Task 3: Synthesize answer
            synthesis_task = Task(
                description=f"""Create a clear, direct answer for: {question}
                Requirements:
                - Answer the question directly and completely
                - Include specific details (numbers, timeframes, conditions)
                - Be concise but comprehensive
                - Format as a clear, readable response""",
                expected_output="A clear, complete answer to the question",
                agent=self.synthesizer,
                context=[research_task]
            )
            
            # Task 4: Validate answer
            validation_task = Task(
                description=f"""Validate the synthesized answer for: {question}
                Check for accuracy, completeness, and proper formatting.
                If improvements are needed, provide the corrected version.""",
                expected_output="Final validated answer",
                agent=self.validator,
                context=[synthesis_task]
            )
            
            # Execute crew
            crew = Crew(
                agents=[self.query_analyzer, self.researcher, self.synthesizer, self.validator],
                tasks=[analysis_task, research_task, synthesis_task, validation_task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            return result.raw.strip()
            
        except Exception as e:
            return f"Error processing question: {str(e)}"

# Initialize enhanced QA system
qa_system = EnhancedQACrew()

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# API Endpoints
@app.get("/")
async def root():
    return {"status": "healthy", "service": "HackRX Enhanced RAG API"}

@app.get("/health")
async def health_check():
    try:
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "vector_count": stats.get('total_vector_count', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/hackrx/run", response_model=HackRXResponse)
async def process_hackrx_request(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """
    Enhanced HackRX endpoint with improved RAG accuracy
    """
    start_time = time.time()
    
    try:
        # Process document
        chunk_count = await rag_system.process_document(request.documents)
        
        # Process questions with enhanced RAG
        answers = []
        
        for i, question in enumerate(request.questions):
            try:
                answer = qa_system.answer_question(question)
                answers.append(answer)
            except Exception as e:
                answers.append(f"Error processing question: {str(e)}")
        
        # Cleanup vector store
        try:
            index.delete(delete_all=True)
        except:
            pass
        
        processing_time = time.time() - start_time
        
        return HackRXResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)