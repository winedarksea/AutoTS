"""Private NumPy least-squares utilities with explicit diagonal penalties."""

import numpy as np


def _penalty_diagonal(penalty, n_features):
    if penalty is None:
        return np.zeros(n_features)
    penalty = np.broadcast_to(np.asarray(penalty, dtype=float), (n_features,))
    if not np.all(np.isfinite(penalty)) or np.any(penalty < 0):
        raise ValueError("Linear penalties must be finite and nonnegative.")
    return penalty


def _augmented_design(design, penalty):
    return np.vstack((design, np.diag(np.sqrt(penalty))))


def solve_linear_coefficients(design, targets, penalty=None):
    """Solve finite, shared-design least squares, preserving a diagonal ridge loss.

    Targets may be one- or two-dimensional. Missing observations must be removed
    before calling: multiplying NaNs by zero does not remove them from products.
    """
    design = np.asarray(design, dtype=float)
    targets = np.asarray(targets, dtype=float)
    penalty = _penalty_diagonal(penalty, design.shape[1])
    if not np.any(penalty):
        return np.linalg.lstsq(design, targets, rcond=None)[0]
    system = design.T @ design + np.diag(penalty)
    try:
        coefficients = np.linalg.solve(system, design.T @ targets)
        if np.all(np.isfinite(coefficients)):
            return coefficients
    except np.linalg.LinAlgError:
        pass
    # Solve the original penalized loss rather than squaring its condition
    # number again by applying least squares to a singular Gram matrix.
    augmented_targets = np.concatenate(
        (targets, np.zeros((design.shape[1],) + targets.shape[1:])), axis=0
    )
    return np.linalg.lstsq(
        _augmented_design(design, penalty), augmented_targets, rcond=None
    )[0]


def linear_fit_covariance(design, penalty=None):
    """Return unit-noise coefficient covariance and residual degrees of freedom.

    SVD of the design (augmented for ridge) retains the rank convention of
    lstsq without inferring rank from the less stable squared singular values.
    No time-by-time hat matrix is constructed.
    """
    design = np.asarray(design, dtype=float)
    penalty = _penalty_diagonal(penalty, design.shape[1])
    regularized = np.any(penalty)
    augmented = _augmented_design(design, penalty) if regularized else design
    _, singular_values, right_vectors = np.linalg.svd(augmented, full_matrices=False)
    cutoff = np.finfo(float).eps * max(augmented.shape) * singular_values[0]
    keep = singular_values > cutoff
    inverse_root = right_vectors[keep].T / singular_values[keep]
    if not regularized:
        covariance = inverse_root @ inverse_root.T
        return covariance, max(design.shape[0] - int(keep.sum()), 1)
    inverse_system = inverse_root @ inverse_root.T
    gram = design.T @ design
    covariance = inverse_system @ gram @ inverse_system.T
    hat_trace = np.sum(inverse_system * gram.T)
    hat_squared_trace = np.sum(covariance * gram.T)
    residual_dof = design.shape[0] - 2 * hat_trace + hat_squared_trace
    return (covariance + covariance.T) / 2, max(float(residual_dof), 1.0)


def observation_mask_groups(design, targets):
    """Yield (usable rows, target columns), sharing solves across equal masks."""
    valid = np.isfinite(targets) & np.isfinite(design).all(axis=1)[:, None]
    if valid.all():
        yield np.ones(design.shape[0], dtype=bool), np.arange(targets.shape[1])
        return
    packed = np.packbits(valid, axis=0).T
    _, representatives, inverse = np.unique(
        packed, axis=0, return_index=True, return_inverse=True
    )
    for group, representative in enumerate(representatives):
        yield valid[:, representative], np.flatnonzero(inverse == group)
