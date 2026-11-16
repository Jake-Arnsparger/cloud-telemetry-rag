# cloud-telemetry-rag

Python RAG pipeline built on AWS Bedrock and S3 that indexes hardware datasheets and fault logs so you can ask natural-language diagnostic questions and get grounded answers with source references.

**Hardware and Software:** Python 3.11+, `boto3`, `python-dotenv`, FAISS / NumPy, AWS S3, and AWS Bedrock (Titan Text Embeddings V2 & Bedrock Converse API) in `us-east-2`.

**How it works:**
- **Document Storage:** Robotics datasheets, pinout notes, and troubleshooting logs are stored in an Amazon S3 bucket (`docs/`).
- **Ingestion & Indexing:** `ingest.py` reads the files, splits them into overlapping text chunks, generates vector embeddings via Amazon Titan Embeddings V2, and saves a local vector index.
- **Query & Response:** `query_rag.py` embeds a user question, pulls the closest matching chunks via similarity search, and sends that context to AWS Bedrock Converse to generate a grounded troubleshooting answer.

**How to Run:**
1. Clone the repo and set up the virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
2. Create a .env file in the root folder with your AWS credentials and bucket details:
   - AWS_ACCESS_KEY_ID,
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION=us-east-2
   - S3_BUCKET_NAME.
3. Run python ingest.py to build the vector index from your telemetry docs.
4. Run python query_rag.py to ask diagnostic questions against the indexed files.

Credits: Jake Arnsparger
