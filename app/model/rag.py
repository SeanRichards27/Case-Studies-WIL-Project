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
    
    @staticmethod
    def extract_citations(answer):
        return list(dict.fromkeys(
            re.findall(r"\b(?:RG|MS)_P\d{2}\b", answer)
        ))
    
    def citation_details(self, answer):
        citation_ids = self.extract_citations(answer)
        citations = []

        for chunk_id in citation_ids:
            match = self.chunks[self.chunks['chunk_id'] == chunk_id]
            if match.empty:
                continue

            row = match.iloc[0]
            page = int(row['page'])
            
            citations.append({
                'chunk_id': chunk_id,
                'document_name': row['document_name'],
                'page': page,
            })
        
        return citations

    def generate(self, question, context):
        prompt = f"""
        You are answering a question about Victorian rental rights using retrieved source material.

        Read ALL of the retrieved chunks before answering. Relevant evidence may appear anywhere in the context.

        Rules:
        - Answer using only information explicitly supported by the retrieved context.
        - If any retrieved chunk directly supports the answer, use that evidence and answer the question.
        - Do not refuse simply because some retrieved chunks are irrelevant.
        - Answer all parts of the question that are supported by the retrieved evidence.
        - Do not invent, assume or infer rules that are not stated in the context.
        - Cite the supporting chunk ID after each factual statement, for example [RG_P24].
        - Only cite chunks that actually support the statement.
        - If none of the retrieved chunks contain enough information to answer the question, respond exactly:
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
        citations = self.citation_details(answer)
        return answer, citations