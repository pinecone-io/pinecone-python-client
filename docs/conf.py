from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Pinecone"
author = "Pinecone"
release = "9.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "myst_parser",
]

html_theme = "furo"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "pinecone/db_data.py",
    "pinecone/db_data/**",
    "pinecone/db_control/**",
    "pinecone/admin/resources/**",
    "pinecone/config/**",
    "pinecone/utils/response_info.py",
    "pinecone_plugins/**",
    "README.md",
]

autodoc_mock_imports = ["pinecone._grpc"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}

autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_returns = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]

copybutton_prompt_text = r">>> |\.\.\. "

suppress_warnings = ["myst.header", "intersphinx"]
