# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'navground_learning'
copyright = '2024, Jerome Guzzi et al. (IDSIA, USI-SUPSI)'
author = 'Jerome Guzzi et al. (IDSIA, USI-SUPSI)'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'nbsphinx',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['Thumbs.db', '.DS_Store']

# autodoc_typehints_format = 'short'
# autodoc_member_order = 'groupwise'
# autodoc_class_signature = 'mixed'
autodoc_inherit_docstrings = False
autoclass_content = 'class'
autodoc_docstring_signature = True
autodoc_typehints = "both"
autodoc_type_aliases = {}

intersphinx_mapping = {
    'gymnasium': ('https://gymnasium.farama.org', None),
    'navground': ('https://idsia-robotics.github.io/navground/_build/html', None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
