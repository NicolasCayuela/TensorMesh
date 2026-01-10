# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os 
import sys 
import datetime 
sys.path.insert(0, os.path.abspath('../..'))
import tensormesh    

project = 'tensormesh'
author  = 'walkerchi'
copyright = f'{datetime.datetime.now().year}, {author}'
version = tensormesh.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
     'sphinx.ext.autodoc',
     'sphinx.ext.napoleon',
     'sphinx.ext.viewcode',
     'sphinx.ext.mathjax',
     'sphinx.ext.githubpages',
     'sphinx.ext.intersphinx',
     'sphinx.ext.autosummary',
     'nbsphinx'
]


html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "text": "TensorMesh",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/walkerchi/tensormesh",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_toc_level": 2,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "collapse_navigation": False,
    # "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"], # Default is good
}

html_sidebars = {
    "**": ["sidebar-nav-bs"]
}

# Remove RTD specific options
# html_logo and html_favicon handled by theme options or below
html_favicon = '_static/logo.png'
html_static_path = ['_static']
templates_path = ['_templates']

add_module_names = False
autodoc_member_order = 'bysource'

suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}

exclude_patterns = []

napoleon_google_docstring = False

autosummary_generate = True

autodoc_member_order = 'groupwise'

# -- Options for HTML output -------------------------------------------------

def skip(app, what, name, obj, skip, options):
     # print(f"what: {what}, name: {name}, obj: {obj}, skip: {skip}, options: {options}\n")
     if hasattr(obj, '__autodoc__'):
          return not name in obj.__autodoc__
     
     return skip

def setup(app):
     app.connect('autodoc-skip-member', skip)
     app.add_css_file('custom.css')

# These settings can also help with documentation structure
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Enable better section numbering
numfig = True
numfig_secnum_depth = 2