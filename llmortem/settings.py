import os
from pathlib import Path

REPO_ROOT = Path(os.getenv("REPO_ROOT", ".")).resolve()

DOCS_DIR = Path(os.getenv("DOCS_DIR", str(REPO_ROOT / "docs")))
SRE_DOCS_DIR = Path(os.getenv("SRE_DOCS_DIR", str(DOCS_DIR / "sre")))
RUNBOOK_DIR = Path(os.getenv("RUNBOOK_DIR", str(REPO_ROOT / "runbooks")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(REPO_ROOT / "index_data")))

REPO_BROWSER_BASE_URL = os.getenv("REPO_BROWSER_BASE_URL", "").rstrip("/")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.28.144.1:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))

MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "220"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "35"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".kt", ".rs", ".rb",
    ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".scala", ".sh", ".sql",
}

DOC_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".rst", ".yaml", ".yml", ".json", ".html",
    ".log", ".conf", ".ini", ".service", ".sh",
}

EXCLUDED_DIR_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", "node_modules",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", "index_data",
}