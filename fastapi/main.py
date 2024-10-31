from fastapi import FastAPI, HTTPException, Query, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
import snowflake.connector
from dotenv import load_dotenv
import boto3
import os
import logging
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
# from llama_index import ServiceContext
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# S3 configuration
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
)
BUCKET_NAME = os.getenv('BUCKET_NAME')

# JWT configuration
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 16

# Initialize OCR model globally (only once for better performance)
ocr_model = PaddleOCR()

# FastAPI app
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Initialize Pinecone using the specified ServerlessSpec format
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

INDEX_NAME = "vec-db-ertsfy"
if INDEX_NAME not in pc.list_indexes():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # Ensure this matches your embedding dimensions
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(INDEX_NAME)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize NVIDIA model settings for embeddings and LLMs
embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
llm_model = NVIDIA(model="meta/llama-3.1-70b-instruct")
# service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm_model)

# Define the Pinecone-based vector store for indexing and querying
class PineconeVectorStore:
    def __init__(self, index):
        self.index = index

    def upsert(self, vectors):
        upserts = [(id, embedding) for id, embedding, _ in vectors]
        self.index.upsert(upserts)

    def query(self, vector, top_k: int):
        results = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return results["matches"]

# Initialize the Pinecone vector store for custom operations
pinecone_vector_store = PineconeVectorStore(index=index)

# Utility function for JWT token creation
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

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

# Helper functions for user authentication
def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        if row:
            return {"username": row[0], "password": row[1], "created_at": row[2]}
        return None
    except Exception as e:
        logging.error(f"Error fetching user: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def create_user(username: str, password: str):
    hashed_password = get_password_hash(password)
    created_at = datetime.utcnow()
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO users (username, password, created_at)
            VALUES (%s, %s, %s)
        """, (username, hashed_password, created_at))
        connection.commit()
    except Exception as e:
        logging.error(f"Error creating user: {e}")
    finally:
        cursor.close()
        connection.close()

# Endpoint to sign up a new user
@app.post("/signup")
async def signup(username: str = Query(...), password: str = Query(...)):
    if get_user(username):
        raise HTTPException(status_code=400, detail="Username already registered")
    create_user(username, password)
    return {"message": "User created successfully"}

# Endpoint to login and generate a JWT token
@app.post("/login", response_model=Token)
async def login(username: str = Query(...), password: str = Query(...)):
    user = get_user(username)
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

# Function to generate embeddings using NVIDIA model
def generate_embedding(text: str):
    try:
        return service_context.embed_model.encode([text])[0]
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

# Process PDF from S3
@app.post("/process-pdf-from-s3/")
async def process_pdf_from_s3(request: PDFRequest):
    pdf_key = request.pdf_key

    try:
        s3_object = s3_client.get_object(Bucket=BUCKET_NAME, Key=pdf_key)
        pdf_data = s3_object['Body'].read()
        logging.info(f"Successfully fetched PDF from S3: {pdf_key}")
    except Exception as e:
        logging.error(f"Error fetching PDF from S3: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching PDF from S3: {e}")

    try:
        pages = convert_from_bytes(pdf_data, 300)
        if not pages:
            logging.error("PDF conversion returned no pages.")
            raise HTTPException(status_code=500, detail="PDF conversion returned no pages.")
        image_paths = []

        for i, page in enumerate(pages):
            image_path = f"temp_image_{i}.png"
            page.save(image_path, "PNG")
            image_paths.append(image_path)

            ocr_results = ocr_model.ocr(image_path)

            embeddings = []
            metadata = []
            if ocr_results:
                for result in ocr_results:
                    text = result[-1][0] if result else ""
                    if text:
                        embedding = generate_embedding(text)
                        embeddings.append(embedding)
                        metadata.append({"pdf_key": pdf_key, "page": i + 1, "text": text})

            # Check if embeddings were generated before attempting to upsert
            if embeddings:
                try:
                    index.upsert(
                        vectors=[(f"{pdf_key}_page_{i + 1}", embedding) for embedding in embeddings],
                        namespace='documents'
                    )
                    logging.info(f"Stored embeddings for page {i + 1} in Pinecone index")
                except Exception as insert_exception:
                    logging.error(f"Error inserting embeddings into Pinecone: {insert_exception}")
                    raise HTTPException(status_code=500, detail="Failed to insert embeddings into Pinecone")
            else:
                logging.warning(f"No embeddings generated for page {i + 1}, skipping upsert.")

        return {"message": "PDF processed successfully", "images": image_paths}

    except Exception as e:
        logging.error(f"Error converting PDF pages to images: {e}")
        raise HTTPException(status_code=500, detail=f"Error converting PDF pages to images: {e}")


# Q/A Endpoint to retrieve and answer queries using NVIDIA model
@app.post("/query/")
async def query_documents(request: QueryRequest):
    query = request.query
    try:
        query_embedding = generate_embedding(query)

        results = pinecone_vector_store.query(
            vector=query_embedding,
            top_k=5
        )

        combined_text = " ".join([res["metadata"]["text"] for res in results])
        answer = Settings.llm.generate_response(query, context=combined_text)
        return {"query": query, "answer": answer}

    except Exception as e:
        logging.error(f"Error querying documents: {e}")
        raise HTTPException(status_code=500, detail="Error querying documents")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
