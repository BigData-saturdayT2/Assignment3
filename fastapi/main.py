from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import snowflake.connector
from dotenv import load_dotenv
import boto3
import os

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

# JWT and security configurations (hardcoded)
SECRET_KEY = "your_secret_key"  # Replace with a strong key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15

# FastAPI app
app = FastAPI()

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
        query = f"SELECT * FROM users WHERE username = '{username}'"
        cursor.execute(query)
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
    




# Snowflake connection configuration using environment variables
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA')
}

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
