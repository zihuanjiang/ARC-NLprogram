"""
Entrypoint for ARC experiment runner.
Delegates to arc.cli for Hydra-based configuration.
"""
from arc.cli import main

if __name__ == "__main__":
    main()
