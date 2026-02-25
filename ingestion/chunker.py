import json
from typing import List
from utils.datatypes import DocumentChunk, Modality
from utils.logger import logger
from utils.schema import DrugInfo

class HealthcareChunker:
    """
    Memory-efficient section-based chunking.
    No overlapping tokens, keeps each chunk strictly bound to a single section
    from the structured JSON, ensuring chunks are lightweight (<400 tokens typically).
    """

    def chunk_json_file(self, file_path: str) -> List[DocumentChunk]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            drug_info = DrugInfo(**data)
            
            source = drug_info.source
            drug_name = drug_info.drug_name
            
            sections = {
                "uses": drug_info.uses,
                "dosage_info": drug_info.dosage_info,
                "common_side_effects": drug_info.common_side_effects,
                "serious_side_effects": drug_info.serious_side_effects,
                "contraindications": drug_info.contraindications,
                "warnings": drug_info.warnings,
                "drug_class": drug_info.drug_class
            }
            
            doc_chunks = []
            for idx, (section_name, content) in enumerate(sections.items()):
                if not content.strip():
                    continue
                    
                metadata = {
                    "drug": drug_name,
                    "section": section_name,
                    "source": source,
                    "chunk_index": idx
                }
                
                text_content = f"Drug: {drug_name}\nSection: {section_name.replace('_', ' ').title()}\nInformation: {content}"
                
                doc_chunk = DocumentChunk(
                    source_file=file_path,
                    modality=Modality.TEXT,
                    text_content=text_content,
                    metadata=metadata
                )
                doc_chunks.append(doc_chunk)
                
            return doc_chunks
                
        except Exception as e:
            logger.error(f"Error chunking healthcare document {file_path}: {e}")
            return []
