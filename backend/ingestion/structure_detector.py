"""
Document Structure Detector — Identifies headings, lists, and paragraphs
from preprocessed Markdown-like text.
"""
import re
from typing import List

from backend.datatypes import DocumentPage, DocumentSection
from backend.logger import logger


class StructureDetector:
    """Detect document structure: headings, lists, paragraphs, and section boundaries."""

    def detect_sections(self, pages: List[DocumentPage]) -> List[DocumentSection]:
        """Parse pages into structured sections."""
        all_sections = []

        for page in pages:
            sections = self._parse_page(page)
            all_sections.extend(sections)

        if not all_sections:
            for page in pages:
                all_sections.append(
                    DocumentSection(
                        title="Content",
                        content=page.text,
                        level=1,
                        page_number=page.page_number,
                    )
                )

        logger.info(f"Detected {len(all_sections)} sections across {len(pages)} pages")
        return all_sections

    def _parse_page(self, page: DocumentPage) -> List[DocumentSection]:
        """Parse a single page into sections based on headings."""
        sections = []
        lines = page.text.split("\n")
        doc_name = page.metadata.get("document_name", "unknown")

        current_title = ""
        current_level = 1
        current_content_lines = []

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if heading_match:
                # Flush previous section
                if current_content_lines or current_title:
                    content = "\n".join(current_content_lines).strip()
                    if content or current_title:
                        sections.append(
                            DocumentSection(
                                title=current_title or "Untitled",
                                content=content,
                                level=current_level,
                                page_number=page.page_number,
                            )
                        )

                # Start new section
                current_level = len(heading_match.group(1))
                current_title = heading_match.group(2).strip()
                current_content_lines = []
            else:
                current_content_lines.append(line)

        # Flush final section
        content = "\n".join(current_content_lines).strip()
        if content or current_title:
            sections.append(
                DocumentSection(
                    title=current_title or "Content",
                    content=content,
                    level=current_level,
                    page_number=page.page_number,
                )
            )

        # Annotate section metadata
        for section in sections:
            section_meta = {
                "document_name": doc_name,
                "page_number": page.page_number,
            }

        return sections

    def detect_element_type(self, text: str) -> str:
        """Classify a text block as heading, list, or paragraph."""
        stripped = text.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            return "heading"
        if re.match(r"^[\-\*\•]\s+", stripped):
            return "list"
        if re.match(r"^\d+[\.\)]\s+", stripped):
            return "ordered_list"
        return "paragraph"
