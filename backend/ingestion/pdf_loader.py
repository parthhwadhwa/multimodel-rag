"""
PDF Loader — Extracts text and formatting metadata from PDFs using PyMuPDF.
"""
import os
from typing import List

import fitz  # PyMuPDF

from backend.utils.datatypes import DocumentPage
from backend.utils.logger import logger


class PDFLoader:
    """Load PDFs and extract page-level text with formatting metadata."""

    def load(self, file_path: str) -> List[DocumentPage]:
        """Load a single PDF and return a list of DocumentPage objects."""
        if not os.path.exists(file_path):
            logger.error(f"PDF not found: {file_path}")
            return []

        pages = []
        try:
            doc = fitz.open(file_path)
            doc_name = os.path.basename(file_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

                page_text = ""
                formatting_spans = []

                for block in blocks:
                    if block["type"] != 0:  # text block only
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span["text"]
                            font = span.get("font", "")
                            size = span.get("size", 10)
                            flags = span.get("flags", 0)

                            is_bold = bool(flags & 2**4) or "Bold" in font or "bold" in font
                            is_italic = bool(flags & 2**1) or "Italic" in font or "italic" in font

                            span_info = {
                                "text": text,
                                "font": font,
                                "size": round(size, 1),
                                "bold": is_bold,
                                "italic": is_italic,
                                "start": len(page_text),
                                "end": len(page_text) + len(text),
                            }
                            formatting_spans.append(span_info)
                            page_text += text

                        page_text += "\n"
                    page_text += "\n"

                if page_text.strip():
                    pages.append(
                        DocumentPage(
                            page_number=page_num + 1,
                            text=page_text.strip(),
                            metadata={
                                "document_name": doc_name,
                                "file_path": file_path,
                                "total_pages": len(doc),
                                "formatting_spans": formatting_spans,
                            },
                        )
                    )

            doc.close()
            logger.info(f"Loaded {len(pages)} pages from {doc_name}")

        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")

        return pages

    def load_directory(self, directory: str) -> List[DocumentPage]:
        """Load all PDFs from a directory."""
        all_pages = []
        pdf_files = sorted(
            [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
        )

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return all_pages

        logger.info(f"Loading {len(pdf_files)} PDFs from {directory}...")
        for pdf_file in pdf_files:
            file_path = os.path.join(directory, pdf_file)
            pages = self.load(file_path)
            all_pages.extend(pages)

        logger.info(f"Total pages loaded: {len(all_pages)}")
        return all_pages
