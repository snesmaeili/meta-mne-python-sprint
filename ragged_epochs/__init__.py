"""Prototype: variable-duration epochs for MNE-Python.

Raggedness is a data representation; temporal alignment is an analysis
decision. The two are separate layers here and never collapse into each other.

  layer 1  RaggedEpochs        trials at their true durations
  layer 2  map_epochs / ops    per-trial operations, no kernel rewrites
  layer 3  align_time          explicit common-coordinate transformations
"""

from ._backends import BACKENDS, AwkwardStore, ListStore, PaddedStore, RaggedStore
from ._container import RaggedEpochs, RaggedTimesError
from ._provenance import AlignmentRecord

__version__ = "0.1.0.dev0"

__all__ = [
    "RaggedEpochs",
    "RaggedTimesError",
    "AlignmentRecord",
    "RaggedStore",
    "ListStore",
    "PaddedStore",
    "AwkwardStore",
    "BACKENDS",
]
