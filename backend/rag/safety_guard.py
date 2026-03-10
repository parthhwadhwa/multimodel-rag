"""
Safety Guard — Jailbreak prevention, prompt injection detection, and input sanitization.
"""
import re
from typing import Tuple

from backend.utils.logger import logger


# Patterns that indicate jailbreak or prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|rules?|prompts?)",
    r"disregard\s+(all\s+)?(previous|above|prior)",
    r"forget\s+(everything|all)\s+(you|that)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+(a|an|if)\s+",
    r"pretend\s+(you|to\s+be)",
    r"new\s+instructions?\s*:",
    r"system\s*prompt\s*:",
    r"override\s+(your|the|all)\s+(rules?|instructions?|constraints?)",
    r"do\s+not\s+follow\s+(your|the)\s+(rules?|guidelines?)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"bypass\s+(safety|filter|guard|restriction)",
]

# Topics that are outside the medical document domain
OFF_TOPIC_PATTERNS = [
    r"\b(write|generate|create)\s+(a\s+)?(poem|story|song|essay|joke|code|script)\b",
    r"\b(hack|exploit|attack|crack|break\s+into)\b",
    r"\b(how\s+to\s+make|synthesize|manufacture)\s+(a\s+)?(bomb|weapon|drug|explosive)\b",
    r"\bhow\s+to\s+(harm|kill|hurt|injure)\b",
]

REFUSAL_MESSAGE = (
    "I'm designed to provide information only from the medical documents in my knowledge base. "
    "I cannot follow instructions that ask me to bypass my guidelines or discuss topics "
    "outside of medication and healthcare information. Please ask a question about "
    "medications, dosages, side effects, or drug interactions."
)


class SafetyGuard:
    """Input validation, jailbreak detection, and prompt injection prevention."""

    def __init__(self):
        self._injection_re = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]
        self._offtopic_re = [
            re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS
        ]

    def check(self, query: str) -> Tuple[bool, str]:
        """
        Validate a user query.
        Returns (is_safe, message).
        If not safe, message contains the refusal reason.
        """
        if not query or not query.strip():
            return False, "Please provide a question."

        sanitized = self._sanitize(query)

        # Check for prompt injection
        for pattern in self._injection_re:
            if pattern.search(sanitized):
                logger.warning(f"Prompt injection detected: {query[:100]}")
                return False, REFUSAL_MESSAGE

        # Check for off-topic requests
        for pattern in self._offtopic_re:
            if pattern.search(sanitized):
                logger.warning(f"Off-topic request detected: {query[:100]}")
                return False, REFUSAL_MESSAGE

        # Length check
        if len(sanitized) > 2000:
            return False, "Query is too long. Please keep your question under 2000 characters."

        return True, ""

    def _sanitize(self, text: str) -> str:
        """Basic input sanitization."""
        # Remove potential HTML/script tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove null bytes
        text = text.replace("\x00", "")
        # Normalize excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def sanitize_output(self, response: str) -> str:
        """Sanitize LLM output to prevent any leaked system instructions."""
        # Remove any attempted system prompt leaks
        sensitive_patterns = [
            r"(?i)system\s*prompt\s*:",
            r"(?i)my\s+instructions\s+are",
            r"(?i)I\s+was\s+told\s+to",
        ]
        for pattern in sensitive_patterns:
            response = re.sub(pattern, "[REDACTED]", response)
        return response
