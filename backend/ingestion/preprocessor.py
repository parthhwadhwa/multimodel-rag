"""
Text Preprocessor — Cleans and normalizes extracted PDF text,
preserving formatting cues as Markdown.
"""
import re
from typing import List, Dict, Any

from backend.datatypes import DocumentPage
from backend.logger import logger


class TextPreprocessor:
    """Clean and normalize PDF-extracted text into Markdown-like format."""

    def process(self, pages: List[DocumentPage]) -> List[DocumentPage]:
        """Process a list of pages, returning cleaned versions."""
        processed = []
        for page in pages:
            cleaned_text = self._clean_text(page.text)
            formatted_text = self._apply_formatting(
                cleaned_text, page.metadata.get("formatting_spans", [])
            )
            processed.append(
                DocumentPage(
                    page_number=page.page_number,
                    text=formatted_text,
                    metadata={
                        k: v
                        for k, v in page.metadata.items()
                        if k != "formatting_spans"
                    },
                )
            )
        return processed

    def _clean_text(self, text: str) -> str:
        """Remove noise and normalize whitespace."""
        # Remove null bytes and control characters (except newlines)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize unicode whitespace
        text = re.sub(r"\u00a0", " ", text)
        # Collapse multiple spaces (preserve newlines)
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return text.strip()

    def _apply_formatting(
        self, text: str, spans: List[Dict[str, Any]]
    ) -> str:
        """Convert formatting spans into Markdown annotations."""
        if not spans:
            return text

        lines = text.split("\n")
        formatted_lines = []
        heading_sizes = self._detect_heading_sizes(spans)

        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append("")
                continue

            # Check if the line content corresponds to a heading-sized span
            level = self._get_heading_level(stripped, spans, heading_sizes)
            if level > 0:
                prefix = "#" * level
                formatted_lines.append(f"{prefix} {stripped}")
            else:
                formatted_lines.append(stripped)

        return "\n".join(formatted_lines)

    def _detect_heading_sizes(self, spans: List[Dict[str, Any]]) -> Dict[str, int]:
        """Identify distinct font sizes used as headings."""
        sizes = set()
        for span in spans:
            if span.get("bold") and len(span.get("text", "").strip()) > 0:
                sizes.add(span["size"])

        sorted_sizes = sorted(sizes, reverse=True)
        mapping = {}
        for i, size in enumerate(sorted_sizes[:3]):
            mapping[str(size)] = i + 1  # h1, h2, h3
        return mapping

    def _get_heading_level(
        self, line_text: str, spans: List[Dict[str, Any]], heading_sizes: Dict[str, int]
    ) -> int:
        """Check if a line text matches a bold span at a heading size."""
        for span in spans:
            span_text = span.get("text", "").strip()
            if span_text and span_text in line_text and span.get("bold"):
                size_key = str(span.get("size", 0))
                if size_key in heading_sizes:
                    return heading_sizes[size_key]
        return 0
