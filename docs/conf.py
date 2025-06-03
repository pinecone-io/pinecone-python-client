import pinecone

project = "Pinecone Python SDK"
author = "Pinecone Systems, Inc."
version = pinecone.__version__
copyright = "%Y, Pinecone Systems, Inc."

html_baseurl = "https://sdk.pinecone.io/python"
html_static_path = ["_static"]
html_favicon = "favicon-32x32.png"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc.typehints",
    "myst_parser",
]

# -- HTML Configuration -------------------------------------------------

html_theme = "alabaster"
html_theme_options = {
    "logo": "pinecone-logo.svg",
    "description": "Pinecone Python SDK",
    "github_user": "pinecone-io",
    "github_repo": "pinecone-python-client",
    "github_button": True,
    "fixed_sidebar": True,
    "page_width": "1140px",
    "sidebar_width": "300px",
    "show_related": False,
    "show_powered_by": False,
    "extra_nav_links": {
        "Github Source": "https://github.com/pinecone-io/pinecone-python-client",
        "Pinecone Home": "https://pinecone.io",
        "Pinecone Docs": "https://docs.pinecone.io",
        "Pinecone Console": "https://app.pinecone.io",
    },
}
