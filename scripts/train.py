#!/usr/bin/env python3
"""
DEPRECATED: This file is a legacy wrapper.

Use the unified training script instead:
    python train.py [options]

This wrapper exists only for backwards compatibility.
It simply forwards all arguments to the root train.py.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import and run the unified trainer
if __name__ == "__main__":
    print("=" * 60)
    print("WARNING: Using legacy scripts/train.py wrapper")
    print("Please use 'python train.py' instead (root directory)")
    print("=" * 60)
    print()

    # Import main from root train.py
    from train import main
    main()
