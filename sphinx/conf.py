import pinecone

project = "Pinecone Python SDK"
author = "Pinecone Systems, Inc."
version = pinecone.__version__

html_baseurl = "https://sdk.pinecone.io/python"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc.typehints",
]

# -- HTML Configuration -------------------------------------------------

html_theme = "alabaster"
html_theme_options = {
    "github_user": "pinecone-io",
    "github_repo": "pinecone-python-client",
    "github_button": True,
    "fixed_sidebar": True,
    "page_width": "1140px",
    "show_related": True,
    # 'analytics_id': '', # TODO: add analytics id
    "description": version,
    "show_powered_by": False,
}
