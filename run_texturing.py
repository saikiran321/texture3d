#!/usr/bin/env python3
"""Back-compat shim: texturing now lives in the installable `texture3d` package.

Equivalent to the `texture3d` console command:

    pip install -e .
    texture3d apt0_seq0 [--mesh reconstructions/apt0_seq0/mesh.ply] ...
"""

from texture3d.cli import main

if __name__ == "__main__":
    main()
