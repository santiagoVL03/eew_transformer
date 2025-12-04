"""
Compatibility module for loading external checkpoints.

This module exists to allow loading checkpoints that were saved
with a different module structure (e.g., from external repositories).
"""

from eew.model import Transformer


class TFEQ:
    """
    Compatibility class for loading external checkpoints.
    This is an alias for the Transformer class.
    """
    # Create a nested TFEQ class for the unpickler
    class TFEQ(Transformer):
        """Nested TFEQ class that inherits from Transformer."""
        pass
