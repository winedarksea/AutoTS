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


def solve_huber_coefficients(design, targets, penalty=None, iterations=50):
    """Fit frozen-MAD Huber regression across all targets sharing usable rows.

    Each update uses matrix products over the full target group. Only the
    dependent optimization iterations and distinct observation masks require
    Python loops; no loop or chunking is performed over individual targets.
    Temporary residual arrays scale with observations times targets.
    """
    if iterations < 1:
        raise ValueError("Huber iterations must be positive.")
    design = np.asarray(design, dtype=float)
    targets = np.asarray(targets, dtype=float)
    single_target = targets.ndim == 1
    if single_target:
        targets = targets[:, None]
    coefficients = np.empty((design.shape[1], targets.shape[1]))
    penalty_diagonal = _penalty_diagonal(penalty, design.shape[1])
    for rows, columns in observation_mask_groups(design, targets):
        x = design[rows]
        if not rows.any():
            raise ValueError("Huber fitting requires observations for every series.")
        gram = x.T @ x + np.diag(penalty_diagonal)
        try:
            operator = np.linalg.solve(gram, x.T)
        except np.linalg.LinAlgError:
            augmented = _augmented_design(x, penalty_diagonal)
            operator = np.linalg.pinv(augmented)[:, : len(x)]
        y = targets[np.ix_(rows, columns)]
        current = solve_linear_coefficients(x, y, penalty_diagonal)
        residual = y - x @ current
        center = np.median(residual, axis=0)
        scale = 1.4826 * np.median(np.abs(residual - center), axis=0)
        floor = 100 * np.finfo(float).eps * np.maximum(1, np.max(np.abs(y), axis=0))
        threshold = 1.345 * np.maximum(scale, floor)
        base = operator @ y
        clipped_residual = np.empty_like(residual)
        for _ in range(iterations):
            # Reuse panel-sized buffers to bound memory when fitting wide data.
            np.matmul(x, current, out=residual)
            np.subtract(y, residual, out=residual)
            np.clip(residual, -threshold, threshold, out=clipped_residual)
            np.subtract(residual, clipped_residual, out=residual)
            current = base - operator @ residual
        coefficients[:, columns] = current
    return coefficients[:, 0] if single_target else coefficients
