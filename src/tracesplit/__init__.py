"""TraceSplit research prototype. No remote service is used during execution."""
__version__ = "0.1.0"
from .language import compile_source, LanguageError
from .engine import run_source, run_plan, save_result
__all__ = ["compile_source", "LanguageError", "run_source", "run_plan", "save_result"]
