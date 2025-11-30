import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pymc as pm
import xarray as xr
import arviz as az
import utils  # your own module

from scipy import stats as stats
from matplotlib import pyplot as plt
from matplotlib import style
from colorama import Fore
from watermark import watermark

import pytensor
pytensor.config.cxx = "/usr/bin/clang++"


# ---- PATHS / STYLE ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLE_FILE = PROJECT_ROOT / "statistical-rethinking-pymc.mplstyle"


def _print_imports():
    """Print the imported module aliases in a readable way."""
    print(Fore.RED + f"Module aliases imported by init_notebook.py:\n{'-' * 44}")
    # Just list the main aliases instead of parsing the file
    aliases = {
        "np": np,
        "pd": pd,
        "smf": smf,
        "pm": pm,
        "xr": xr,
        "az": az,
        "stats": stats,
        "plt": plt,
        "utils": utils,
    }
    for name in aliases:
        print(Fore.GREEN + "import" + Fore.BLUE + f" {name}")


def _print_watermark():
    print(Fore.RED + f"Watermark:\n{'-' * 10}")
    print(Fore.BLUE + watermark())
    print(Fore.BLUE + watermark(iversions=True, globals_=globals()))


def _set_warnings_and_style():
    warnings.filterwarnings("ignore")

    if STYLE_FILE.exists():
        style.use(STYLE_FILE)
        print(Fore.BLUE + f"Using style file: {STYLE_FILE}")
    else:
        print(Fore.RED + f"Style file not found at: {STYLE_FILE}")


def setup():
    """
    Configure notebook environment:
    - set PyTensor compiler
    - ignore warnings
    - apply matplotlib style
    - print imports and watermark
    """
    _set_warnings_and_style()
    _print_imports()
    _print_watermark()