"""
LangSmith Integration - Enables explicit tracing for the LangGraph orchestrator.
"""
import os
from backend.utils.config import CONFIG
from backend.utils.logger import logger

def setup_langsmith_tracing():
    """Configure environment variables to auto-activate LangSmith telemtry."""
    if CONFIG.langsmith.tracing_enabled and CONFIG.langsmith.api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = CONFIG.langsmith.api_key
        os.environ["LANGCHAIN_PROJECT"] = CONFIG.langsmith.project_name
        logger.info(f"LangSmith Tracing Enabled for project: {CONFIG.langsmith.project_name}")
    else:
        # Explicitly disable if there's no API key to avoid annoying warnings
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith Tracing is Disabled (no API key or explicitly false)")
