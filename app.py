import os
import streamlit as st
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import boto3
import tempfile
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Fetch the NVIDIA API key and other configurations from the environment
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY is not loaded. Please check your .env file.")

# Set the NVIDIA API key in the environment for the LangChain NVIDIA endpoint library
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

# Initialize NVIDIA LLM
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", max_tokens=1024)

# AWS S3 configurations
BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_PDFS_FOLDER = os.getenv("S3_PDFS_FOLDER")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# List PDFs in S3
def list_pdfs_from_s3():
    print("Listing PDF files in S3 bucket.")
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PDFS_FOLDER)
    pdfs = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".pdf")]
    print(f"Found PDFs in S3: {pdfs}")
    return pdfs

# Load a PDF from S3
def load_pdf_from_s3(bucket_name: str, s3_key: str):
    print(f"Loading PDF {s3_key} from S3 bucket {bucket_name}.")
    pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    pdf_content = pdf_obj["Body"].read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_pdf.write(pdf_content)
        tmp_pdf_path = tmp_pdf.name

    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()
    os.remove(tmp_pdf_path)
    
    print(f"Loaded {len(documents)} document(s) from the PDF.")
    return documents

# Streamlit UI
def show_process_pdf_page():
    MAX_CONTEXT_LENGTH = 2000  # Decrease chunk size to allow for more chunks

    st.subheader("Process PDF Document")
    pdf_files = list_pdfs_from_s3()
    selected_pdf = st.selectbox("Select a PDF Document to Process", pdf_files)

    if selected_pdf and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            documents = load_pdf_from_s3(BUCKET_NAME, selected_pdf)

            # Combine all document pages into a single context string
            context = "\n".join([doc.page_content for doc in documents])

            # Split context into manageable chunks
            context_chunks = [context[i:i + MAX_CONTEXT_LENGTH] for i in range(0, len(context), MAX_CONTEXT_LENGTH)]
            st.session_state['context_chunks'] = context_chunks
            st.write("PDF processed successfully. Click the button below to get a summary of the entire document.")

    # Only show the summarize button if context has been processed
    if 'context_chunks' in st.session_state:
        if st.button("Generate Summary"):
            context_chunks = st.session_state['context_chunks']

            # Iterate through chunks and generate individual summaries
            combined_summary = []  # To store the summaries from each chunk
            for chunk_index, chunk in enumerate(context_chunks):
                prompt = f"Please summarize the following context in 6 lines:\n\nContext (Chunk {chunk_index + 1}/{len(context_chunks)}): {chunk}"

                print(f"Generated Prompt for Chunk {chunk_index + 1}:\n{prompt}")

                # Attempt to get the response using NVIDIA API
                try:
                    response = llm.invoke(prompt)  # Directly pass the prompt as a string
                    if not response:
                        print(f"No answer was generated for Chunk {chunk_index + 1}. Please try a different question or check logs for issues.")
                        st.write(f"No answer was generated for Chunk {chunk_index + 1}. Please try a different question or check logs for issues.")
                    else:
                        print(f"Received Response for Chunk {chunk_index + 1}: {response}")
                        # Store the summary for each chunk
                        combined_summary.append(response.content)
                except Exception as e:
                    print(f"Failed to retrieve response from NVIDIA API for Chunk {chunk_index + 1}. Error: {e}")
                    st.write(f"An error occurred while fetching the response for Chunk {chunk_index + 1}. Please try again.")
                    continue  # Continue processing other chunks even if an error occurs

            # Combine all chunk summaries into one large context for final summarization
            if combined_summary:
                full_summary_context = "\n".join(combined_summary)
                final_prompt = f"Please summarize the following combined context in 6 lines:\n\n{full_summary_context}"

                # Generate the final summary
                try:
                    final_response = llm.invoke(final_prompt)
                    if not final_response:
                        print("No final summary was generated. Please try again.")
                        st.write("No final summary was generated. Please try again.")
                    else:
                        print(f"Final Summary: {final_response}")
                        st.write("Summary:", final_response.content)
                except Exception as e:
                    print(f"Failed to generate final summary. Error: {e}")
                    st.write("An error occurred while generating the final summary. Please try again.")

# Main Streamlit App
def main():
    st.title("NVIDIA RAG Document Exploration App")
    menu_options = ["Process and Summarize PDF"]

    with st.sidebar:
        choice = st.selectbox("Menu", menu_options)

    if choice == "Process and Summarize PDF":
        show_process_pdf_page()

if __name__ == "__main__":
    main()
