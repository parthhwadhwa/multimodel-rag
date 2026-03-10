"""
LangChain adapters for the Document Intelligence System.
Safely converts internal datatypes to LangChain documents.
"""
from typing import List, Dict, Any, Union
from langchain_core.documents import Document

from backend.utils.datatypes import DocumentChunk, RetrievalResult, DocumentPage

def to_langchain_document(item: Union[DocumentChunk, RetrievalResult, DocumentPage]) -> Document:
    """Convert a native datatype to a LangChain Document."""
    if isinstance(item, RetrievalResult):
        # Flatten score into metadata
        metadata = item.chunk.metadata.copy()
        metadata["score"] = item.score
        metadata["retrieval_method"] = item.retrieval_method
        metadata["parent_id"] = item.chunk.parent_id
        
        return Document(
            page_content=item.chunk.text,
            metadata=metadata
        )
        
    elif isinstance(item, DocumentChunk):
        metadata = item.metadata.copy()
        metadata["parent_id"] = item.parent_id
        return Document(
            page_content=item.text,
            metadata=metadata
        )
        
    elif isinstance(item, DocumentPage):
        metadata = item.metadata.copy()
        metadata["page_number"] = item.page_number
        return Document(
            page_content=item.text,
            metadata=metadata
        )
        
    else:
        raise ValueError(f"Unsupported item type for conversion: {type(item)}")

def to_langchain_documents(items: List[Union[DocumentChunk, RetrievalResult, DocumentPage]]) -> List[Document]:
    """Bulk convert native datatypes to LangChain Documents."""
    return [to_langchain_document(item) for item in items]
