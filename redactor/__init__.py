"""Top-level package for contract redaction toolkit.

Exports the Pipeline and Span types for convenient imports.
"""
from .pipeline import Pipeline, Span, MaskStyle, PipelineOptions

__all__ = ["Pipeline", "Span", "MaskStyle", "PipelineOptions"]
