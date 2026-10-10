"""Exceptions raised by SeeDL."""


class SeeDLError(Exception):
    """Base exception for an invalid or unsupported SeeDL operation."""


class UnsupportedObjectError(SeeDLError, TypeError):
    """Raised when :func:`seedl.see` cannot select a visualization for an object."""


class MissingDependencyError(SeeDLError, ImportError):
    """Raised when an explicitly requested optional renderer is unavailable."""
