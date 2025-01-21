import numpy as np
import numpy.typing as npt

from typing import Sequence, Set, Tuple


def check_dims(
    polytopes: Sequence[np.ndarray],
):
    if not all(len(polytope.shape) == 2 for polytope in polytopes):
        raise ValueError(
            "Polytope representations should be provided as 2-dimensional NumPy arrays."
        )
    dim = max(polytope.shape[1] for polytope in polytopes)
    if not all(polytope.shape[1] == dim for polytope in polytopes):
        raise ValueError(
            "Polytope representations should have matching second dimension."
        )
    return dim


def _sample_vertices_prealloc(
    num_samples: int,
    polytopes: Sequence[npt.ArrayLike],
    upper_hull: bool = False,
):
    polytope_arr = list(np.array(polytope) for polytope in polytopes)
    # Ensure all V-representations live in the same dimension
    dim = check_dims(polytope_arr)
    # Preallocate Gaussian samples
    samples = np.random.randn(num_samples, dim)
    # Flip samples with negative first element
    if upper_hull:
        neg_indices = samples[:, 0] < 0
        samples[neg_indices, :] = -samples[neg_indices, :]
    # Preallocate vertex list and index list
    upper_hull_inds = np.zeros(shape=(len(polytope_arr), num_samples), dtype=np.int64)
    lower_hull_inds = np.zeros(shape=(len(polytope_arr), num_samples), dtype=np.int64)
    for polytope_idx, polytope in enumerate(polytope_arr):
        inner_products = polytope @ samples.T
        upper_hull_inds[polytope_idx, :] = np.argmax(
            inner_products,
            axis=0,
        )
        lower_hull_inds[polytope_idx, :] = np.argmin(
            inner_products,
            axis=0,
        )
    # The following properties hold:
    # 1. Every column of upper_hull_inds contains a tuple of maximizing indices;
    # 2. Every column of lower_hull_inds contains a tuple of minimizing indices;
    # 3. There is one column for each (conditional) Gaussian sample.
    vertex_inds = set(
        tuple(upper_hull_inds[:, idx].tolist()) for idx in range(num_samples)
    )
    if not upper_hull:
        vertex_inds.update(
            tuple(lower_hull_inds[:, idx].tolist()) for idx in range(num_samples)
        )
    return vertex_inds


def sample_convex_hull_vertices(
    num_samples: int,
    polytopes: Sequence[npt.ArrayLike],
    upper_hull: bool = False,
    prealloc_samples: bool = False,
) -> Set[Tuple[int]]:
    """Sample vertices from the convex hull of a Minkowski sum.

    This method implements Algorithm 1 from [1]. It assumes every polytope is
    provided via its V-representation (i.e., a list of vertices). This method
    may underestimate the number of vertices, especially if `num_samples` is
    too small. It returns a set of tuples, with each tuple corresponding to a
    different vertex in the convex hull of the Minkowski sum.

    [1]: https://arxiv.org/abs/1805.08749

    Args:
        num_samples (int): The number of Gaussian samples to use.
        polytopes (Sequence[numpy.typing.ArrayLike]): An iterable containing
            the V-representation of each polytope. Each V-representation should
            be a 2-dimensional NumPy array, with the number of columns equal to
            the ambient dimension.
        upper_hull (bool): Set to only count vertices in the upper hull.
        prealloc_samples (bool): Set to preallocate the Gaussian samples. If not
            set, samples will be processed sequentially.

    Returns:
        A set of indices corresponding to vertices in the (upper) convex hull.

    Raises:
        ValueError: If the polytope representation is invalid.
    """
    if prealloc_samples:
        return _sample_vertices_prealloc(
            num_samples,
            polytopes,
            upper_hull,
        )
    polytope_arr = list(np.array(polytope) for polytope in polytopes)
    # Ensure all V-representations live in the same dimension
    dim = check_dims(polytope_arr)
    # Preallocate vertex list and index list
    upper_hull_inds = np.zeros(shape=(len(polytope_arr), num_samples), dtype=np.int64)
    lower_hull_inds = np.zeros(shape=(len(polytope_arr), num_samples), dtype=np.int64)
    for sample_idx in range(num_samples):
        sample = np.random.randn(dim)
        # Flip sample if first element is negative
        if upper_hull and sample[0] < 0:
            sample = -sample
        for polytope_idx, polytope in enumerate(polytope_arr):
            inner_product = polytope @ sample
            upper_hull_inds[polytope_idx, sample_idx] = np.argmax(inner_product)
            lower_hull_inds[polytope_idx, sample_idx] = np.argmin(inner_product)
    # The following properties hold:
    # 1. Every column of upper_hull_inds contains a tuple of maximizing indices;
    # 2. Every column of lower_hull_inds contains a tuple of minimizing indices;
    # 3. There is one column for each (conditional) Gaussian sample.
    vertex_inds = set(
        tuple(upper_hull_inds[:, idx].tolist()) for idx in range(num_samples)
    )
    if not upper_hull:
        vertex_inds.update(
            tuple(lower_hull_inds[:, idx].tolist()) for idx in range(num_samples)
        )
    return vertex_inds
