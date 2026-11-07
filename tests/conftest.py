"""Pytest configuration for SeeDL test suite.

Ensures non-GUI matplotlib backend is configured before any matplotlib imports.
"""

import os

# Ensure matplotlib uses the non-GUI Agg backend in all test runs
os.environ["MPLBACKEND"] = "Agg"

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

try:
    import plotly.io as pio

    pio.renderers.default = None
except ImportError:
    pass
