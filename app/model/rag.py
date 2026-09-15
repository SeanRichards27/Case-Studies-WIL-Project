import pandas as pd
import requests
import re

from rank_bm25 import BM25Okapi
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

CURRENT_DIR = Path.cwd()
if (CURRENT_DIR / "data" / "processed").exists():
    DATA_DIR = CURRENT_DIR / "data" / "processed"
elif (CURRENT_DIR.parent / "data" / "processed").exists():
    DATA_DIR = CURRENT_DIR.parent / "data" / "processed"
else:
    raise FileNotFoundError("Could not locate data/processed folder")

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

class RentalRAG:
    def __init__(self, data_dir = DATA_DIR):
        self.chunks = pd.read_json(data_dir / "rental_kb_chunks.jsonl",lines=True)

        self.test_collection = pd.read_json(data_dir / "rental_test_collection_v1.jsonl",lines=True)

        self._bm25 = BM25Okapi(self.chunks["text"].apply(tokenize).tolist())

    def retrieve_bm25(self, question, top_k = 5):
        query_tokens = tokenize(question)
        scores = self._bm25.get_scores(query_tokens)

        results = self.chunks.copy()
        results["bm25_score"] = scores

        return (
            results
            .sort_values("bm25_score", ascending=False)
            .head(top_k)[
                ["chunk_id", "document_name", "page", "bm25_score", "text"]
            ]
            .reset_index(drop=True)
        )

    @staticmethod
    def get_context(retrived):
        context_parts = []
        for _, row in retrived.iterrows():
            context_parts.append(
                f"[{row['chunk_id']}] "
                f"{row['document_name']} - Page {row['page']}\n"
                f"{row['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        return context

    def generate(self, question, context):
        prompt = f"""
        You are answering questions about Victorian rental rights.

        Use ONLY the retrieved context provided below.
        Do not use outside knowledge or make assumptions.

        Instructions:
        - Answer the user's question clearly and concisely.
        - Support factual statements using the relevant chunk ID in square brackets, for example [RG_P24].
        - Only cite chunks that actually support the statement.
        - If the retrieved context does not contain enough information to answer the question, say:
        "The retrieved information does not contain enough information to answer this question."

        RETRIEVED CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            }
        )

        response.raise_for_status()

        return response.json()["response"].strip()
    
    def ask(self, question, top_k = 5):
        retrived = self.retrieve_bm25(question, top_k)
        answer = self.generate(question, self.get_context(retrived))

        return answer