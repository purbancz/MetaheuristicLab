"""Sparse coordinate-wise role PSO algorithms."""

from algorithm.sparse_roles.coordinate_mask_utilities import CoordinateMaskMixin, SparseCoordinateMixin
from algorithm.sparse_roles.sparse_role_based import (
    SparseAmnesiacPSO,
    SparseAnarchicAmnesiacPSO,
    SparseAnarchicPSO,
    SparseContrarianPSO,
    SparseContrarianDefeatistPSO,
    SparseDefeatistPSO,
    SparseDrifterPSO,
    SparseErraticPSO,
    SparseEscapistPSO,
    SparseEschewerEscapistPSO,
    SparseEschewerPSO,
    SparseRebelPSO,
    SparseRebelRejectorPSO,
    SparseRejectorPSO,
    SparseWandererPSO,
)
from algorithm.sparse_roles.sparse_hybrid import (
    SparseHybridAdditivePSO,
    SparseHybridFullDisjointPSO,
    SparseHybridPartialDisjointPSO,
)

__all__ = [
    "CoordinateMaskMixin",
    "SparseCoordinateMixin",
    "SparseWandererPSO",
    "SparseDefeatistPSO",
    "SparseContrarianDefeatistPSO",
    "SparseRebelPSO",
    "SparseRejectorPSO",
    "SparseRebelRejectorPSO",
    "SparseContrarianPSO",
    "SparseEschewerPSO",
    "SparseEscapistPSO",
    "SparseEschewerEscapistPSO",
    "SparseAnarchicPSO",
    "SparseAmnesiacPSO",
    "SparseAnarchicAmnesiacPSO",
    "SparseErraticPSO",
    "SparseDrifterPSO",
    "SparseHybridPartialDisjointPSO",
    "SparseHybridFullDisjointPSO",
    "SparseHybridAdditivePSO",
]
