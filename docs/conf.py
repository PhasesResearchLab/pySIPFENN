# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pysipfenn.core.pysipfenn import __version__, __file__

project = 'pySIPFENN'
copyright = '2023, Adam M. Krajewski'
author = 'Adam M. Krajewski'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.linkcode',
              'sphinx.ext.duration',
              'sphinx.ext.napoleon',
              #'sphinx_autodoc_typehints',
              'myst_parser',
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
exclude_dirnames = ["../pysipfenn/tests"]

napoleon_use_param = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

html_context = {
    "display_github": True,
    "github_user": "PhasesResearchLab",
    "github_repo": "pySIPFENN",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

def linkcode_resolve_simple(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/PhasesResearchLab/pySIPFENN/tree/main/%s.py" % filename

# Resolve function for the linkcode extension.
# Thanks to https://github.com/Lasagne/Lasagne/blob/master/docs/conf.py
def linkcode_resolve(domain, info):
    def find_source():
        import inspect
        import os, sys
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None

    try:
        rel_path, line_start, line_end = find_source()
        # __file__ is imported from pysipfenn.core.pysipfenn
        filename = f"pysipfenn/core/{rel_path}#L{line_start}-L{line_end}"
    except Exception:
        # no need to be relative to core here as module includes full path.
        filename = info["module"].replace(".", "/") + ".py"

    tag = "v" + __version__
    return f"https://github.com/PhasesResearchLab/pySIPFENN/blob/{tag}/{filename}"