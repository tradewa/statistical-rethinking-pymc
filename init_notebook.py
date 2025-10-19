
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pymc as pm
import xarray as xr
import arviz as az
import utils as utils

from scipy import stats as stats
from matplotlib import pyplot as plt

import pytensor
pytensor.config.cxx = '/usr/bin/clang++'


# ---
# Print imports / aliases
with open(__file__) as f:
    lines = f.readlines()

from colorama import Fore
def print_import(import_line):
    stripped = import_line.strip()
    if not stripped or stripped.startswith("#"):
        return
    
    parts = stripped.replace(",", " ").split()
    if not parts:
        return
    
    if parts[0] == 'import':
        module = parts[1]
        msg = Fore.GREEN + 'import' + Fore.BLUE + f" {module}"
        if "as" in parts[2:]:
            alias = parts[parts.index("as", 2) + 1]
            msg += Fore.GREEN + " as" + Fore.BLUE + f" {alias}"
    elif parts[0] == "from" and "import" in parts:
        module = parts[1]
        import_idx = parts.index("import")
        if import_idx + 1 >= len(parts):
            return
        submodule = parts[import_idx + 1]
        msg = (
            Fore.GREEN + "from" + Fore.BLUE + f" {module} "
            + Fore.GREEN + "import" + Fore.BLUE + f" {submodule}"
        )
        if "as" in parts[import_idx + 2:]:
            alias_idx = parts.index("as", import_idx + 2)
            if alias_idx + 1 < len(parts):
                alias = parts[alias_idx + 1]
                msg += Fore.GREEN + " as" + Fore.BLUE + f" {alias}"
    else:
        return

    print(msg)

print(Fore.RED + f"Module aliases imported by init_notebook.py:\n{'-'* 44}")
for l in lines:
    if "# ---" in l:
        break
    print_import(l)

from watermark import watermark
print(Fore.RED + f"Watermark:\n{'-'* 10}")
print(Fore.BLUE + watermark())
print(Fore.BLUE + watermark(iversions=True, globals_=globals()))


import warnings
warnings.filterwarnings("ignore")

from matplotlib import style
STYLE = "statistical-rethinking-pymc.mplstyle"
style.use(STYLE)