import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import snowflake.connector
from dotenv import load_dotenv
import boto3
import os
import uuid
import logging
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from io import BytesIO
import fitz  # PyMuPDF for PDF processing

# Pinecone and NVIDIA imports
from pinecone import Pinecone, ServerlessSpec
# Local imports from your project
from processor import get_pdf_documents
from utils import set_environment_variables
from llama_index.embeddings.nvidia import NVIDIAEmbedding
import pinecone
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os




# Load environment variables
load_dotenv()

# S3 configuration
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)
BUCKET_NAME = os.getenv('BUCKET_NAME')
S3_PDFS_FOLDER = os.getenv("S3_PDFS_FOLDER")

# JWT configuration
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 16

# Initialize OCR model globally (only once for better performance)
ocr_model = PaddleOCR()

# FastAPI app
app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        INDEX_NAME = "kkkkkkk"
        DIMENSION = 1024

        # Check if the index exists; if not, create it
        if INDEX_NAME not in pc.list_indexes():
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        global index
        index = pc.Index(INDEX_NAME)
        print("Pinecone index initialized successfully")

    except Exception as e:
        print(f"Error initializing Pinecone in FastAPI: {e}")

@app.get("/")
async def root():
    return {"message": "Hello World"}


class PineconeVectorStore:
    def __init__(self, index):
        self.index = index

    def upsert(self, vectors):
        try:
            # Prepare the data in the required format for Pinecone's upsert method
            upserts = [{"id": id, "values": embedding, "metadata": metadata} for id, embedding, metadata in vectors]
            self.index.upsert(upserts)
            logging.info("Successfully upserted vectors into Pinecone.")
        except Exception as e:
            logging.error(f"Error during upsert operation in Pinecone: {e}")
            raise HTTPException(status_code=500, detail="Error upserting data to Pinecone")

    def query(self, vector, top_k: int):
        try:
            # Perform the query with the given vector and top_k value
            results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
            logging.info("Successfully queried Pinecone index.")
            return results["matches"]
        except Exception as e:
            logging.error(f"Error during query operation in Pinecone: {e}")
            raise HTTPException(status_code=500, detail="Error querying data from Pinecone")
    

# Initialize NVIDIA model settings for embeddings and LLMs
embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")

# Define Pinecone-based vector store for indexing and querying
class PineconeVectorStore:
    def __init__(self, index):
        self.index = index

    def upsert(self, vectors):
        upserts = [(id, embedding) for id, embedding, _ in vectors]
        self.index.upsert(upserts)

    def query(self, vector, top_k: int):
        results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return results["matches"]


# Models for requests
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserProfile(BaseModel):
    username: str
    created_at: datetime

class UpdatePassword(BaseModel):
    old_password: str
    new_password: str

class PDFRequest(BaseModel):
    pdf_key: str

class QueryRequest(BaseModel):
    query: str

# Database connection
def get_db_connection():
    return snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )


# Function to generate embeddings using NVIDIA model
def generate_embedding(text: str):
    try:
        return embed_model.encode([text])[0]
    except AttributeError:
        logging.error("The NVIDIAEmbedding model does not support the 'encode' method. Check method compatibility.")
        raise HTTPException(status_code=500, detail="Embedding model does not support 'encode'")

# Endpoint to list PDFs from S3
@app.get("/s3/pdfs")
async def list_pdfs():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix='pdfs1/')
        files = [obj['Key'] for obj in response.get('Contents', [])]
        return {"files": files}
    except Exception as e:
        logging.error(f"Error listing PDFs from S3: {e}")
        raise HTTPException(status_code=500, detail="Error listing PDFs from S3")

# Endpoint to retrieve data from PUBLICATION_DATA table in Snowflake
@app.get("/publications")
async def get_publications():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = "SELECT TITLE, BRIEF_SUMMARY, IMAGE_LINK, PDF_LINK FROM PUBLICATION_DATA"
        cursor.execute(query)
        rows = cursor.fetchall()

        publications = [
            {"title": row[0], "brief_summary": row[1], "image_link": row[2], "pdf_link": row[3]}
            for row in rows
        ]

        return {"publications": publications}

    except Exception as e:
        logging.error(f"Error retrieving publications from Snowflake: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving publications")
    finally:
        cursor.close()
        connection.close()


# Endpoint to list all PDF files in S3
@app.get("/list-pdfs/")
async def list_pdfs():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PDFS_FOLDER)
        pdf_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]
        return {"pdf_files": pdf_files}
    except Exception as e:
        print(f"Error listing PDFs from S3: {e}")
        raise HTTPException(status_code=500, detail="Error listing PDFs from S3")

# Load NVIDIA API Key from environment variables
NVIDIA_API_KEY = os.getenv("NGC_API_KEY")

# Define request model
class TextEmbeddingRequest(BaseModel):
    text: str

@app.post("/get-embedding/")
async def get_embedding(request: TextEmbeddingRequest):
    """
    FastAPI endpoint to get embeddings from NVIDIA's cloud API.
    """
    url = "https://api.ngc.nvidia.com/v2/text/embedding"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": [request.text]}

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])[0]
            return {"embedding": embedding}

        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to retrieve embedding")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
