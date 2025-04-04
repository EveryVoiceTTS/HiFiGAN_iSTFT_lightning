"""
This file turns the package into a module that can be run as a script with
    python -m hfgl ...
or, for coverage analysis, with
    coverage run -m hfgl ...
"""

from .cli import app

app()
