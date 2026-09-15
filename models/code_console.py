"""
Lightweight, per-dataset "notebook kernel" that powers the Code tab.

Each active dataset (keyed by its filepath, same key used by app.py's
DATASET_STACKS) gets its own persistent execution namespace so variables
created in one cell are still available in the next -- the same mental
model as a Jupyter/Colab kernel. The namespace is seeded with the dataset
already loaded into `df` (reflecting whatever preprocessing has been
applied so far), plus pandas/numpy/matplotlib preloaded for convenience.

NOTE: this executes arbitrary Python via exec()/eval() with the process's
full builtins. That is an intentional trade-off for a single-user, local
data-analysis tool that wants Colab-like freedom (importing extra
libraries, plotting, etc). It is NOT hardened for a multi-tenant/public
deployment -- a production build of this feature should run each kernel
in an isolated subprocess/container with resource + time limits instead
of exec()'ing directly inside the Flask worker.
"""

import ast
import base64
import builtins
import contextlib
import io
import traceback

import matplotlib
matplotlib.use("Agg")  # headless rendering -- no GUI backend available on the server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# filepath -> {"ns": dict, "exec_count": int}
CODE_KERNELS = {}


def _seed_namespace(df):
    """Builds a fresh kernel namespace with the dataset + common libraries preloaded."""
    return {
        "__builtins__": builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df.copy(),
    }


def get_kernel(filepath, df_provider):
    """Returns the existing kernel for this dataset, creating one (seeded with
    the current dataset) if it doesn't exist yet."""
    kernel = CODE_KERNELS.get(filepath)
    if kernel is None:
        kernel = {"ns": _seed_namespace(df_provider()), "exec_count": 0}
        CODE_KERNELS[filepath] = kernel
    return kernel


def reset_kernel(filepath, df_provider):
    """Wipes all kernel state and re-imports the dataset fresh (used on first
    open of the Code tab and on an explicit 'Restart Kernel' click)."""
    kernel = {"ns": _seed_namespace(df_provider()), "exec_count": 0}
    CODE_KERNELS[filepath] = kernel
    return kernel


def drop_kernel(filepath):
    """Frees kernel memory for a dataset that's being replaced/deleted."""
    CODE_KERNELS.pop(filepath, None)


def _capture_figures():
    """Grabs every open matplotlib figure as a base64 PNG, then closes them
    so the next cell run doesn't keep re-rendering old plots."""
    images = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            images.append(base64.b64encode(buf.read()).decode("utf-8"))
        except Exception:
            continue
    plt.close("all")
    return images


def _render_value(value):
    """Renders a cell's trailing expression the way a notebook would --
    DataFrames/Series as an HTML table, everything else as repr()."""
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return {
            "type": "html",
            "content": value.to_html(
                classes="table table-sm table-striped table-hover mb-0", index=True
            ),
        }
    if isinstance(value, pd.Series):
        return {
            "type": "html",
            "content": value.to_frame().to_html(
                classes="table table-sm table-striped table-hover mb-0"
            ),
        }
    return {"type": "text", "content": repr(value)}


def execute_code(filepath, code, df_provider):
    """Runs one cell's code inside the dataset's persistent kernel namespace.

    Mirrors Jupyter's "last bare expression is auto-displayed" behavior, and
    captures stdout, stderr, matplotlib figures, and tracebacks separately so
    the frontend can render them in the right order/style.
    """
    kernel = get_kernel(filepath, df_provider)
    ns = kernel["ns"]
    kernel["exec_count"] += 1
    exec_count = kernel["exec_count"]

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    outputs = []
    error = None

    try:
        tree = ast.parse(code, mode="exec")

        # If the cell ends in a bare expression (e.g. `df.head()`), pull it
        # out so we can eval + auto-display it like a notebook cell does.
        trailing_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing_expr = tree.body.pop()

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            if tree.body:
                exec(compile(tree, "<cell>", "exec"), ns)

            if trailing_expr is not None:
                value = eval(
                    compile(ast.Expression(trailing_expr.value), "<cell>", "eval"), ns
                )
                rendered = _render_value(value)
                if rendered:
                    outputs.append(rendered)

        for img in _capture_figures():
            outputs.append({"type": "image", "content": img})

    except Exception:
        error = traceback.format_exc()

    return {
        "exec_count": exec_count,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "outputs": outputs,
        "error": error,
    }
