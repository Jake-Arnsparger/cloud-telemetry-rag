import csv
import json
import os
import pickle
import boto3
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-2")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
CSV_PATH = os.path.join("data", "telemetry_run_01.csv")
S3_KEY = "raw-logs/telemetry_run_01.csv"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def upload_to_s3(local_path: str, bucket: str, key: str) -> None:
    s3.upload_file(local_path, bucket, key)
    print(f"[S3] Uploaded {local_path} -> s3://{bucket}/{key}")

def build_chunks(csv_path: str) -> list[str]:
    chunks = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunk = (
                f"Time {row['timestamp_ms']}ms | Subsystem: {row['subsystem']} | "
                f"Status: {row['status_flag']} | Yaw: {row['yaw_deg']} deg, "
                f"Pitch: {row['pitch_deg']} deg, Ultrasonic: {row['ultrasonic_cm']} cm, "
                f"PWM (L/R): {row['motor_left_pwm']}/{row['motor_right_pwm']} | "
                f"Note: {row['diagnostic_note']}"
            )
            chunks.append(chunk)
    return chunks

def get_embedding(text: str) -> np.ndarray:
    body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
    response = bedrock.invoke_model(modelId=EMBED_MODEL_ID, body=body)
    payload = json.loads(response["body"].read())
    return np.array(payload["embedding"], dtype="float32")

def main() -> None:
    if not BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME is missing in .env")

    upload_to_s3(CSV_PATH, BUCKET_NAME, S3_KEY)
    chunks = build_chunks(CSV_PATH)
    print(f"[EMBED] Generating Titan V2 embeddings for {len(chunks)} log entries...")

    vectors = np.vstack([get_embedding(c) for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])  # Inner product on normalized vectors = cosine similarity
    index.add(vectors)

    faiss.write_index(index, "telemetry.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("[DONE] Saved local index (telemetry.faiss) and metadata (chunks.pkl).")


if __name__ == "__main__":
    main()