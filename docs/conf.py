# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
from datetime import date
project = 'AutoTS'
copyright = u'%s, Colin Catlin' % date.today().year
author = 'Colin Catlin'

# The full version, including alpha/beta/rc tags
# import AutoTS
# from  AutoTS import __version__
# release = __version__
release = "0.2.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# Add napoleon to the extensions list
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'm2r', 'sphinx.ext.githubpages']

source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

with open("_googleid.txt","r") as f:
	gid = f.readline().strip()


html_theme_options = {
    "show_powered_by": False,
	'analytics_id':gid,
	'logo': 'autots_logo.png',
	'description': 'Automated Forecasting',
    "github_user": "winedarksea",
    "github_repo": "autots",
    "github_banner": False,
    "show_related": False,
    "note_bg": "#FFF59C",
}
# Output file base name for HTML help builder.
htmlhelp_basename = "autotsdoc"