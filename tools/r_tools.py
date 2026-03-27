from __future__ import annotations

"""Shared R bootstrap helpers used across notebook workflows.

This module centralizes repo-wide utilities that prepare the R environment,
install required R packages, and lazily load `rpy2`. Notebook-specific model
logic should stay in the workflow modules that use these helpers.
"""

import json
import locale
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

__all__ = ["setup_r_environment", "ensure_r_packages", "load_rpy2", "r_literal"]


_RPY2_CALLBACKS_PATCHED = False


def setup_r_environment() -> None:
    """Locate R and configure `R_HOME` and `PATH` for `rpy2` workflows.

    Typical use in this repo is to call this once near the top of a notebook
    before importing helpers that rely on R or `rpy2`.
    """
    if 'R_HOME' not in os.environ:
        # Search common installation directories before requiring manual setup.
        possible_paths = [
            r'C:\Program Files\R\R-4.4.1',
            r'C:\Program Files\R\R-4.4.0',
            r'C:\Program Files\R\R-4.3.3',
            r'C:\Program Files\R\R-4.3.2',
            r'C:\Program Files\R\R-4.3.1',
            r'C:\Program Files\R\R-4.3.0',
            r'C:\Program Files\R\R-4.2.3',
            r'C:\Program Files\R\R-4.2.2',
            r'C:\Program Files\R\R-4.2.1',
            r'C:\Program Files\R\R-4.2.0',
            r'C:\Program Files\R\R-4.1.3',
            r'C:\Program Files (x86)\R\R-4.4.1',
            r'C:\Program Files (x86)\R\R-4.3.3',
            r'C:\Users\%USERNAME%\Documents\R\R-4.4.1',
            '/usr/lib/R',
            '/usr/local/lib/R',
            '/opt/R',
        ]

        r_home = None
        for path in possible_paths:
            expanded_path = os.path.expandvars(path)
            if Path(expanded_path).exists():
                r_home = expanded_path
                break

        if r_home is None:
            raise RuntimeError(
                "R installation not found. Please either:\n"
                "1. Install R from https://cran.r-project.org/\n"
                "2. Set R_HOME environment variable manually\n"
                "3. Add R to your system PATH\n\n"
                f"Searched in the following locations:\n" +
                "\n".join([f"  - {os.path.expandvars(p)}" for p in possible_paths])
            )

        os.environ['R_HOME'] = r_home

        # On Windows, rpy2 typically needs the x64 binary directory on PATH.
        if sys.platform.startswith('win'):
            bin_path = Path(r_home) / 'bin' / 'x64'
        else:
            bin_path = Path(r_home) / 'bin'

        if bin_path.exists():
            os.environ['PATH'] += os.pathsep + str(bin_path)

        print(f"Using R installation at: {r_home}")
    else:
        print(f"Using existing R_HOME: {os.environ['R_HOME']}")


def _get_rscript_path() -> Path:
    """Return the `Rscript` executable after `setup_r_environment()` has run.

    This helper is shared by notebook bootstrap code that installs or verifies
    R packages before model fitting starts.
    """
    if 'R_HOME' in os.environ:
        r_home = Path(os.environ['R_HOME'])
        bin_candidates = [
            r_home / 'bin' / 'x64' / 'Rscript.exe',
            r_home / 'bin' / 'Rscript.exe',
            r_home / 'bin' / 'Rscript',
        ]
        for candidate in bin_candidates:
            if candidate.exists():
                return candidate

    which_rscript = shutil.which('Rscript')
    if which_rscript:
        return Path(which_rscript)

    raise RuntimeError(
        'Rscript could not be found. Run setup_r_environment() before calling ensure_r_packages().'
    )


def ensure_r_packages(
    required_packages: Iterable[str],
    github_packages: dict[str, str] | None = None,
) -> None:
    """Install any missing R packages needed by a notebook workflow.

    Parameters are split into CRAN packages and optional GitHub-backed packages.
    In this repo, the main example is the WR notebook bootstrap cell.
    """
    github_packages = github_packages or {}
    required_packages = list(required_packages)
    rscript_path = _get_rscript_path()

    cran_packages = [pkg for pkg in required_packages if pkg not in github_packages]
    cran_vector = ', '.join(json.dumps(pkg) for pkg in cran_packages)

    install_lines = [
        "options(repos = c(CRAN = 'https://cloud.r-project.org'))",
        'installed <- rownames(installed.packages())',
    ]

    if cran_packages:
        install_lines.extend(
            [
                f'cran_required <- c({cran_vector})',
                'cran_missing <- setdiff(cran_required, installed)',
                'if (length(cran_missing) > 0) install.packages(cran_missing)',
                'installed <- rownames(installed.packages())',
            ]
        )

    for package_name, repo in github_packages.items():
        install_lines.extend(
            [
                f'if (!({json.dumps(package_name)} %in% installed)) {{',
                f'  remotes::install_github({json.dumps(repo)})',
                '}',
                'installed <- rownames(installed.packages())',
            ]
        )

    subprocess.check_call([str(rscript_path), '-e', '\n'.join(install_lines)])


def load_rpy2():
    """Import `rpy2` lazily so bootstrap can happen before R-backed imports.

    Workflow modules call this only when they are ready to hand data to R.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib import callbacks, conversion, openrlib

    global _RPY2_CALLBACKS_PATCHED
    if sys.platform.startswith('win') and not _RPY2_CALLBACKS_PATCHED:
        preferred_encoding = locale.getpreferredencoding(False) or 'utf-8'

        def _decode_console_bytes(cdata, maxlen=None, encoding=None):
            raw = openrlib.ffi.string(cdata) if maxlen is None else openrlib.ffi.string(cdata, maxlen)
            candidate_encodings = (
                encoding,
                preferred_encoding,
                'utf-8',
                'cp949',
                'cp1252',
            )
            for candidate in candidate_encodings:
                if not candidate:
                    continue
                try:
                    return raw.decode(candidate)
                except UnicodeDecodeError:
                    continue
            return raw.decode(preferred_encoding, errors='replace')

        # rpy2's Windows console callback can attempt UTF-8 on locale-encoded R output.
        conversion._cchar_to_str = lambda c, encoding=None: _decode_console_bytes(c, encoding=encoding)
        conversion._cchar_to_str_with_maxlen = (
            lambda c, maxlen, encoding=None: _decode_console_bytes(c, maxlen=maxlen, encoding=encoding)
        )
        if hasattr(callbacks, '_CCHAR_ENCODING'):
            callbacks._CCHAR_ENCODING = preferred_encoding
        _RPY2_CALLBACKS_PATCHED = True

    return ro, default_converter, numpy2ri, localconverter


def r_literal(value: object) -> str:
    """Convert a Python scalar into an inline R literal for `ro.r(...)` calls."""
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return 'NULL'
    return str(value)

