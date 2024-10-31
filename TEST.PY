from pinecone import Pinecone, ServerlessSpec
import os

# Initialize Pinecone with the API key
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

INDEX_NAME = "vec-db-erts"

try:
    # Check if the index already exists
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
        print(f"Index '{INDEX_NAME}' created successfully.")
    else:
        print(f"Index '{INDEX_NAME}' already exists, using the existing index.")

    # Use the existing or newly created index
    index = pc.Index(INDEX_NAME)

except Exception as e:
    print(f"Error initializing Pinecone index: {e}")
    raise
