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

project = 'TensorMesh'
author = 'Shizheng Wen, Mingyuan Chi'
copyright = f'{datetime.datetime.now().year}, TensorMesh Contributors'
version = tensormesh.__version__
release = version

# -- Internationalization ----------------------------------------------------
# Source language; build other languages with `-D language=zh_CN` etc.
language = 'en'
locale_dirs = ['locale/']
gettext_compact = False  # one .po per source .rst file
gettext_uuid = True      # stable message IDs across pot regenerations

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
     'sphinx_design',
     'nbsphinx'
]


html_theme = 'furo'

# Override the default `f"{project} {release} documentation"` so the sidebar
# brand renders as a clean "TensorMesh" — required for the two-color wordmark
# CSS in _static/custom.css to work.
html_title = 'TensorMesh'

html_theme_options = {
    'light_logo': 'logo.png',
    'dark_logo':  'logo.png',
    'sidebar_hide_name': False,

    # Brand palette — overrides Furo's defaults so the whole site picks up the
    # tmblue / tmteal pair from the logo (see logo.tex).
    'light_css_variables': {
        'color-brand-primary': '#5B6EE8',  # tmblue: sidebar active links, TOC highlight
        'color-brand-content': '#149B8E',  # tmteal: in-content anchor links
    },
    'dark_css_variables': {
        'color-brand-primary': '#8B9AFF',
        'color-brand-content': '#2DC9B8',
    },

    # Furo doesn't ship Font Awesome — inline GitHub mark SVG.
    'footer_icons': [
        {
            'name': 'GitHub',
            'url':  'https://github.com/camlab-ethz/TensorMesh',
            'html': (
                '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
                '5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49'
                '-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
                '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2'
                '-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 '
                '0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 '
                '2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07'
                '-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 '
                '.21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>'
                '</svg>'
            ),
            'class': '',
        },
    ],
}

# Sidebar layout: insert language switcher between brand and search.
html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'language-switcher.html',
        'sidebar/search.html',
        'sidebar/scroll-start.html',
        'sidebar/navigation.html',
        'sidebar/scroll-end.html',
    ]
}

html_favicon = '_static/logo.png'
html_static_path = ['_static']
templates_path = ['_templates']

add_module_names = False
autodoc_member_order = 'bysource'

# Preserve source-level expressions for default argument values so
# function-object defaults render as e.g. `strain_fn=strain` rather than
# `strain_fn=<function strain>` (which is just repr() at doc-build time).
autodoc_preserve_defaults = True

# Render parameter type hints with their unqualified ("short") name —
# e.g. `Tensor` instead of `~torch.Tensor` — and hyperlink them via
# intersphinx. Without this, Sphinx 9 leaks the `~` short-name marker
# into the rendered signature for parameters (return types are unaffected).
python_use_unqualified_type_names = True

suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'torch': ('https://pytorch.org/docs/master', None),
}

exclude_patterns = ['example_gallery/_archive/*']

# Custom inline roles available in every .rst file. Used in index.rst to color
# "Tensor" / "Mesh" in the H1 with the brand palette (see _static/custom.css).
rst_prolog = """
.. role:: tensor-blue
.. role:: mesh-teal
"""

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