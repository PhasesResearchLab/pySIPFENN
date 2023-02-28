from pysipfenn.core.pysipfenn import *


def showDocs():
    """Open the offline documentation in a web browser."""
    import os
    if os.path.isfile('docs/_build/index.html'):
        os.system('open docs/_build/index.html')
    else:
        os.system('open https://pysipfenn.org')
        print('Documentation local files were not found. Please be advised that the documentation is only available if'
              'you are in cloned pySIPFENN GitHub repository. pySIPFENN will now attempt to visit '
              'https://pysipfenn.org for the online documentation.')
