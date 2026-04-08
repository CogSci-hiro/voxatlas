Installation
============

VoxAtlas targets Python 3.10 and newer.

Create an environment, install the package, and add any optional extras
needed for the parts of the toolkit you plan to use.

Basic install
-------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

Optional extras
---------------

Install feature-specific dependencies as needed:

.. code-block:: bash

   pip install -e .[acoustic]
   pip install -e .[syntax]
   pip install -e .[dev]

Build the docs locally
----------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs
   make html

The generated site is written to ``docs/_build/html``.
