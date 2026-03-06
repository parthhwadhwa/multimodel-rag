import sys
import os

from unittest.mock import patch
from backend.agents.llm_client import LLMClient
from backend.utils.datatypes import RetrievalResult, DocumentChunk

def test_formatting():
    client = LLMClient()
    
    # Mock documents
    chunk1 = DocumentChunk(text="Metformin lowers blood sugar.", metadata={"document_name": "metformin.pdf", "page_number": 2})
    res1 = RetrievalResult(chunk=chunk1, score=0.85, retrieval_method="hybrid")
    
    chunk2 = DocumentChunk(text="Metformin acts on the liver.", metadata={"document_name": "metformin.pdf", "page_number": 2})
    res2 = RetrievalResult(chunk=chunk2, score=0.75, retrieval_method="hybrid")

    chunk3 = DocumentChunk(text="Used for type 2 diabetes.", metadata={"document_name": "diabetes_drugs.pdf", "page_number": 4})
    res3 = RetrievalResult(chunk=chunk3, score=0.80, retrieval_method="hybrid")
    
    results = [res1, res2, res3]

    class MockResponse:
        def raise_for_status(self): pass
        def json(self): return {"response": "Metformin is a medication used to treat type 2 diabetes. It lowers blood sugar by acting on the liver."}
        def iter_lines(self):
            return [b'{"response":"Metformin "}', b'{"response":"is a medication."}']

    with patch('requests.post', return_value=MockResponse()):
        print("--- GENERATE ---")
        out1 = client.generate("What is metformin?", results)
        print(out1)
        
        print("\n--- GENERATE STREAM ---")
        out2 = list(client.generate_stream("What is metformin?", results))
        print("".join(out2))

if __name__ == "__main__":
    test_formatting()
