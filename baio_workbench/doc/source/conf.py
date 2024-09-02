import os
import sys
from datetime import date

# Set the path to the source code
sys.path.insert(0, os.path.abspath("../../src"))

# Project description
project = "baio"
copyright = f"{date.today().year}, Noah Bruderer"
author = "Noah Bruderer"

# Dynamically retrieve the version from your package
try:
    from check_this import __version__ as release
except ImportError:
    release = "0.0.1"

# Sphinx extension module names
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "docker": ("https://docs.docker.com/get-docker/", None),
    "torchvision": ("https://pytorch.org/docs/master/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Source file suffix
source_suffix = [".rst", ".md"]

# The master toctree document
master_doc = "index"

# Language
language = "en"

# Pygments style
pygments_style = "friendly"

# HTML theme options
html_theme = "furo"
html_title = "Baio"
html_static_path = ["_static"]
html_logo = "_static/images/logo.jpg"

# Custom CSS files
html_css_files = [
    "custom.css",
]

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#007acc",
        "color-brand-content": "#006699",
        "color-admonition-background": "rgba(0, 122, 204, 0.1)",
    },
    "dark_css_variables": {
        "color-brand-primary": "#1d4ed8",
        "color-brand-content": "#1e40af",
        "color-admonition-background": "rgba(29, 78, 216, 0.1)",
    },
    "logo_only": True,  # Only show the logo without the title text
}

# Paths to ignore
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add mappings for viewcode extension
viewcode_import = True
