"""Evals for the research agent.

A full run makes thousands of model calls, so it would use up a LangSmith account's monthly trace quota if
tracing from .env were left on. It is switched off here, before the app is imported (load_dotenv does not
overwrite a variable that is already set). Set EVALS_TRACE=1 to trace an eval run on purpose.
"""

import os

if not os.environ.get("EVALS_TRACE"):
    for _name in ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2"):
        os.environ[_name] = "false"
