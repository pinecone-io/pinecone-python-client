import json
from datetime import datetime
import readline
import os
import atexit


def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        # First try to get a dictionary representation if available
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # Fall back to string representation
        return str(obj)
    except (TypeError, RecursionError):
        # If we hit any serialization issues, return a safe string representation
        return f"<{obj.__class__.__name__} object>"


def install_json_repr_override(klass):
    klass.__repr__ = lambda self: json.dumps(
        self.to_dict(), indent=4, sort_keys=False, default=custom_serializer
    )


def setup_readline_history():
    """Setup readline history for the custom REPL."""
    # Create .pinecone directory in user's home if it doesn't exist
    history_dir = os.path.expanduser("~/.pinecone")
    os.makedirs(history_dir, exist_ok=True)

    # Set up history file
    history_file = os.path.join(history_dir, "repl_history")

    # Load history if it exists
    if os.path.exists(history_file):
        readline.read_history_file(history_file)

    # Set history size
    readline.set_history_length(1000)

    # Save history on exit
    atexit.register(readline.write_history_file, history_file)
