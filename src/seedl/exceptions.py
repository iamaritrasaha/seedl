"""Exceptions raised by SeeDL."""


class SeeDLError(Exception):
    """Base exception for an invalid or unsupported SeeDL operation."""


class UnsupportedObjectError(SeeDLError, TypeError):
    """Raised when :func:`seedl.see` cannot select a visualization for an object."""


class MissingDependencyError(SeeDLError, ImportError):
    """Raised when an explicitly requested optional renderer is unavailable."""


class InvalidSliceError(SeeDLError, ValueError):
    """Raised for unsupported or malformed basic tensor indices."""


class SliceIndexError(SeeDLError, IndexError):
    """Raised for excessive axes or an out-of-bounds tensor index."""
