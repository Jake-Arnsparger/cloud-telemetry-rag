import json
import os
import pickle
import boto3
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-2")
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "us.amazon.nova-micro-v1:0"

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def embed_query(text: str) -> np.ndarray:
    body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
    response = bedrock.invoke_model(modelId=EMBED_MODEL_ID, body=body)
    payload = json.loads(response["body"].read())
    return np.array([payload["embedding"]], dtype="float32")

def retrieve_context(question: str, top_k: int = 3) -> list[str]:
    index = faiss.read_index("telemetry.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    q_vec = embed_query(question)
    scores, indices = index.search(q_vec, top_k)
    return [chunks[i] for i in indices[0] if i != -1]

def ask_diagnostics(question: str) -> str:
    context_lines = retrieve_context(question, top_k=3)
    formatted_context = "\n".join(f"- {line}" for line in context_lines)

    system_prompt = (
        "You are an automated cloud telemetry diagnostic assistant. "
        "Answer strictly using the retrieved telemetry log entries provided. "
        "Cite exact timestamps and status flags. If the logs do not contain the answer, state that clearly."
    )

    user_message = (
        f"Retrieved Telemetry Logs:\n{formatted_context}\n\n"
        f"Diagnostic Question: {question}"
    )

    response = bedrock.converse(
        modelId=LLM_MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0.1},
    )
    return response["output"]["message"]["content"][0]["text"]

if __name__ == "__main__":
    print("=== Cloud Telemetry RAG Diagnostic CLI (type 'exit' to quit) ===")
    while True:
        q = input("\nQuery > ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        print("\n--- Retrieved Evidence & Diagnosis ---")
        print(ask_diagnostics(q))