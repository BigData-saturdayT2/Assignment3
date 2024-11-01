from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from llama_index.embeddings.nvidia import NVIDIAEmbedding
import pinecone
import snowflake.connector
from dotenv import load_dotenv
import requests
import boto3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from document_processors import get_pdf_documents, parse_all_tables, parse_all_images
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from the .env file
load_dotenv()

# Snowflake connection configuration using environment variables
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA')
}

# JWT and security configurations
SECRET_KEY = os.getenv('SECRET_KEY')  # Replace with a strong key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15

EMBD_API_KEY=os.getenv('EMBD_API_KEY')

# Initialize NVIDIA embedding model and Pinecone
nvidia_embedding = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")

# Create an instance of Pinecone
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Define the index name
index_name = 'llm-index'

# Check if the index exists and create it if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches the embedding dimensions
        metric='euclidean',  # Use the metric you need (e.g., cosine, euclidean, etc.)
        spec=ServerlessSpec(
            cloud='aws',  # Specify the cloud provider
            region='us-east-1'  # Specify the region
        )
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# Access the index
index = pc.Index(index_name)

# FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme for JWT tokens
security = HTTPBearer()

# Pydantic models
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

# Snowflake database connection
def get_db_connection():
    return snowflake.connector.connect(
        user=SNOWFLAKE_CONFIG['user'],
        password=SNOWFLAKE_CONFIG['password'],
        account=SNOWFLAKE_CONFIG['account'],
        warehouse=SNOWFLAKE_CONFIG['warehouse'],
        database=SNOWFLAKE_CONFIG['database'],
        schema=SNOWFLAKE_CONFIG['schema']
    )

# Utility functions
def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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
        print(f"Error fetching user: {e}")
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
        print(f"Error creating user: {e}")
    finally:
        cursor.close()
        connection.close()

def embed_and_store_documents(documents, doc_id_prefix):
    vectors = []
    for i, doc in enumerate(documents):
        doc_data = doc.dict() if hasattr(doc, 'dict') else {"text": doc.text, "metadata": doc.metadata}
        
        text = doc_data.get("text")
        metadata = doc_data.get("metadata", {})

        try:
            embedding = nvidia_embedding.embed(text)
            unique_id = f"{doc_id_prefix}-{i}"
            vectors.append((unique_id, embedding, metadata))
        except Exception as e:
            print(f"Error embedding document {i} with prefix {doc_id_prefix}: {e}")

    if vectors:
        index.upsert(vectors)

# API Endpoints
@app.post("/signup")
async def signup(
    username: str = Query(..., description="The username for the new user"),
    password: str = Query(..., description="The password for the new user")
):
    existing_user = get_user(username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    create_user(username, password)
    return {"message": "User created successfully"}

@app.post("/login", response_model=Token)
async def login(
    username: str = Query(..., description="The username of the user"),
    password: str = Query(..., description="The password of the user")
):
    user = get_user(username)
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = "bdiaassignment3"

# Endpoint to list objects in the images1 folder
@app.get("/s3/images")
async def list_images():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix='images1/')
        files = [obj['Key'] for obj in response.get('Contents', [])]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to list objects in the pdfs1 folder
@app.get("/s3/pdfs")
async def list_pdfs():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix='pdfs1/')
        files = [obj['Key'] for obj in response.get('Contents', [])]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to retrieve data from PUBLICATION_DATA table
@app.get("/publications")
async def get_publications():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("USE SCHEMA ASSIGNEMNT_3.PUBLIC")
        query = "SELECT TITLE, BRIEF_SUMMARY, IMAGE_LINK, PDF_LINK FROM PUBLICATION_DATA"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Transform rows into a list of dictionaries
        publications = [
            {"title": row[0], "brief_summary": row[1], "image_link": row[2], "pdf_link": row[3]}
            for row in rows
        ]
        
        return {"publications": publications}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving publications: {str(e)}")
    
    finally:
        cursor.close()
        connection.close()


@app.get("/process_pdf")
async def process_pdf(title: str):
    try:
        # Fetch PDF link from Snowflake
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("USE SCHEMA ASSIGNEMNT_3.PUBLIC")
        cursor.execute("SELECT PDF_LINK FROM PUBLICATION_DATA WHERE TITLE = %s", (title,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="PDF link not found for the given title")

        pdf_link = result[0]
        s3_object_key = pdf_link.replace(f"https://{BUCKET_NAME}.s3.amazonaws.com/", "")

        # Retrieve PDF from S3
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_object_key)
        pdf_content = response['Body'].read()

        # Process the PDF content
        documents = get_pdf_documents(pdf_content)  # Assumes this returns list of chunked docs

        # Embed and store documents in Pinecone
        embed_and_store_documents(documents, title.replace(" ", "_"))

        return {"status": "success", "documents": [doc for doc in documents]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    finally:
        cursor.close()
        connection.close()