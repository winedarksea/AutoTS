# -*- coding: utf-8 -*-
"""Vectorized joint selection of trend hinges (slope changes) and steps (level shifts).

Greedy forward selection under one least-squares model ``[1, t, nuisance,
hinges, steps]`` with a BIC stop, then relocation and pruning. Every candidate
position is scored at once: by Frisch-Waugh-Lovell, adding column ``x`` to a
model with orthonormal basis ``Q`` and residual ``r`` lowers SSE by
``(r.x)^2 / (|x|^2 - |Q'x|^2)``, and for hinge/step columns all the inner
products come from prefix/suffix cumulative sums in O(n) per basis vector.

Conventions: ``t`` is scaled to [0, 1) for conditioning. A step at ``c`` is
``1[i >= c] - 1`` (``-1`` before, ``0`` after), matching LevelShiftMagic, so a
positive coefficient is an upward shift and the final level is unaffected.
All positions and lengths are in points (periods), not days.
"""

import numpy as np

KNOT = 'knot'
STEP = 'step'


def _suffix_sum_after(values):
    """out[c] = sum over i > c of values[i] (row-wise for 2-D)."""
    reverse_cumsum = np.cumsum(values[::-1], axis=0)[::-1]
    out = np.zeros_like(reverse_cumsum)
    out[:-1] = reverse_cumsum[1:]
    return out


def _prefix_sum_before(values):
    """out[c] = sum over i < c of values[i] (row-wise for 2-D)."""
    cumsum = np.cumsum(values, axis=0)
    out = np.zeros_like(cumsum)
    out[1:] = cumsum[:-1]
    return out


class HingeStepCandidates:
    """Closed-form inner products and norms for every hinge/step position."""

    def __init__(self, n):
        self.n = int(n)
        self.t = np.arange(self.n, dtype=float) / self.n
        remaining = (self.n - 1 - np.arange(self.n)).astype(float)
        self.hinge_norm2 = (
            remaining * (remaining + 1) * (2 * remaining + 1) / 6.0 / self.n**2
        )
        self.step_norm2 = np.arange(self.n, dtype=float)

    def dots(self, vectors, kind):
        """Inner products of every candidate column with each column of ``vectors`` -> (n, k)."""
        if kind == KNOT:
            return _suffix_sum_after(vectors * self.t[:, None]) - self.t[
                :, None
            ] * _suffix_sum_after(vectors)
        return -_prefix_sum_before(vectors)

    def norm2(self, kind):
        return self.hinge_norm2 if kind == KNOT else self.step_norm2

    def column(self, position, kind):
        if kind == KNOT:
            return np.maximum(0.0, self.t - self.t[position])
        return (np.arange(self.n) >= position) - 1.0


def _orthonormal_basis(design):
    """Orthonormal basis of the column space; SVD drops collinear nuisance directions
    that a plain QR would keep as spurious extra dimensions."""
    if design.shape[1] == 0:
        return np.zeros((design.shape[0], 0))
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = singular.max() * max(design.shape) * np.finfo(float).eps
    return left[:, singular > tolerance]


def _candidate_gains(candidates, basis, residual, kind, allowed_mask):
    """SSE reduction for adding each candidate position; -inf where not allowed."""
    products = candidates.dots(np.column_stack([residual, basis]), kind)
    residual_products, basis_products = products[:, 0], products[:, 1:]
    denominator = candidates.norm2(kind) - np.einsum(
        "ij,ij->i", basis_products, basis_products
    )
    valid = allowed_mask & (denominator > 1e-12)
    gains = np.full(candidates.n, -np.inf)
    gains[valid] = residual_products[valid] ** 2 / denominator[valid]
    return gains


def _spacing_mask(n, existing_positions, min_gap, low, high):
    """Positions in [low, high] at least ``min_gap`` away from every existing one."""
    positions = np.arange(n)
    mask = (positions >= low) & (positions <= high)
    if len(existing_positions):
        distance = np.abs(positions[:, None] - np.asarray(existing_positions)[None, :])
        mask &= distance.min(axis=1) >= min_gap
    return mask


def _ar1_variance_inflation(residual):
    """(1 + rho) / (1 - rho) so t-tests and windows respect residual autocorrelation."""
    if residual.size < 3 or np.std(residual) <= 0:
        return 1.0
    rho = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    rho = float(np.clip(rho if np.isfinite(rho) else 0.0, 0.0, 0.95))
    return (1.0 + rho) / (1.0 - rho)


class JointTrendModel:
    """Holds the base design and builds full designs for a given term list."""

    def __init__(self, values, nuisance=None):
        self.values = np.asarray(values, dtype=float)
        self.n = self.values.size
        self.candidates = HingeStepCandidates(self.n)
        trend_base = [np.ones(self.n), self.candidates.t]
        self.n_trend_base = len(trend_base)
        nuisance_columns = (
            [np.asarray(nuisance, dtype=float)] if nuisance is not None else []
        )
        self.base = np.column_stack(trend_base + nuisance_columns)
        self._base_basis = None

    def _cached_base_basis(self):
        # The base (intercept, t, Fourier nuisance) is fixed across the hundreds of
        # forward/refine/merge steps, so its SVD is done once; re-SVDing the full
        # n x (base + terms) design per step was ~70% of a joint detector fit.
        if self._base_basis is None:
            left, singular, _ = np.linalg.svd(self.base, full_matrices=False)
            self._base_singular_max = float(singular.max()) if singular.size else 0.0
            tolerance = self._base_singular_max * max(self.base.shape) * np.finfo(float).eps
            self._base_basis = left[:, singular > tolerance]
        return self._base_basis

    def design(self, terms):
        columns = [self.candidates.column(pos, kind) for kind, pos in terms]
        if not columns:
            return self.base
        return np.column_stack([self.base] + columns)

    def basis_and_residual(self, terms):
        """Orthonormal basis of [base, terms] and the residual of values on it.

        Same column space as _orthonormal_basis(self.design(terms)): term columns are
        projected off the cached base basis and only that (n, len(terms)) remainder is
        SVD'd, with the rank tolerance scaled as for the full design.
        """
        base_basis = self._cached_base_basis()
        if terms:
            term_columns = np.column_stack(
                [self.candidates.column(pos, kind) for kind, pos in terms]
            )
            remainder = term_columns
            # twice: one Gram-Schmidt pass leaves base components in near-collinear columns
            for _ in range(2):
                remainder = remainder - base_basis @ (base_basis.T @ remainder)
            left, singular, _ = np.linalg.svd(remainder, full_matrices=False)
            scale = max(
                self._base_singular_max,
                float(np.linalg.norm(term_columns, axis=0).max()),
            )
            n_columns = self.base.shape[1] + len(terms)
            tolerance = scale * max(self.n, n_columns) * np.finfo(float).eps
            basis = np.hstack([base_basis, left[:, singular > tolerance]])
        else:
            basis = base_basis
        residual = self.values - basis @ (basis.T @ self.values)
        return basis, residual

    def term_statistics(self, terms, return_inflation=False):
        """beta, SSE increase if each term is dropped, AR(1)-adjusted t, SSE, variance."""
        design = self.design(terms)
        gram_inverse = np.linalg.pinv(design.T @ design)
        beta = gram_inverse @ (design.T @ self.values)
        residual = self.values - design @ beta
        sse = float(residual @ residual)
        term_index = np.arange(self.base.shape[1], design.shape[1])
        diagonal = np.clip(np.diag(gram_inverse)[term_index], 1e-300, None)
        dof = max(self.n - design.shape[1], 1)
        inflation = _ar1_variance_inflation(residual)
        variance = sse / dof * inflation
        term_beta = beta[term_index]
        drop_cost = term_beta**2 / diagonal
        t_stats = term_beta / np.sqrt(max(variance, 1e-300) * diagonal)
        if return_inflation:
            return beta, drop_cost, t_stats, sse, variance, inflation
        return beta, drop_cost, t_stats, sse, variance


def _merge_closest_pair(
    model, terms, fixed, min_segment, low, high, penalty, sse, inflation
):
    """Replace the closest same-kind pair (< 2 * min_segment apart) by one term if BIC agrees.

    One bent break is often fit as two hinges straddling it; each is expensive to
    drop alone because the survivor cannot move, so test the pair jointly.
    Returns the new term list, or None when no merge is justified.
    """
    best = None
    for kind in (KNOT, STEP):
        positions = sorted(p for k, p in terms if k == kind and (k, p) not in fixed)
        for left, right in zip(positions[:-1], positions[1:]):
            if right - left < 2 * min_segment and (
                best is None or right - left < best[2] - best[1]
            ):
                best = (kind, left, right)
    if best is None:
        return None
    kind, left, right = best
    others = [term for term in terms if term not in ((kind, left), (kind, right))]
    basis, residual = model.basis_and_residual(others)
    mask = np.zeros(model.n, dtype=bool)
    mask[max(low, left) : min(high, right) + 1] = True
    same_kind = [p for k, p in others if k == kind]
    if same_kind:
        distance = np.abs(np.arange(model.n)[:, None] - np.asarray(same_kind)[None, :])
        mask &= distance.min(axis=1) >= min_segment
    gains = _candidate_gains(model.candidates, basis, residual, kind, mask)
    if not np.isfinite(gains).any():
        return None
    position = int(np.argmax(gains))
    merged_sse = float(residual @ residual) - gains[position]
    if model.n * np.log(merged_sse / max(sse, 1e-300)) / inflation < penalty:
        return others + [(kind, position)]
    return None


def select_hinge_step_terms(
    values,
    nuisance=None,
    min_segment=28,
    edge=14,
    max_terms=16,
    bic_multiplier=2.0,
    step_t_threshold=4.0,
    search_steps=True,
    fixed_steps=(),
    max_refine_rounds=5,
    effective_n_prune=True,
):
    """Select hinge knots and steps for one series.

    ``effective_n_prune`` divides the pruning/merge likelihood gain by the AR(1)
    variance inflation of the current residual (BIC on the effective sample size).

    ``fixed_steps`` are externally proposed step positions (e.g. LevelShiftMagic
    candidates): they are never relocated or BIC-pruned, only dropped when their
    joint-model |t| falls below ``step_t_threshold``. ``search_steps`` lets the
    greedy search also propose steps (the unified mode). Returns a sorted list
    of ``(kind, position)``.
    """
    model = JointTrendModel(values, nuisance)
    n = model.n
    if n < max(2 * min_segment, 8) or not np.all(np.isfinite(model.values)):
        return []
    penalty = 2.0 * bic_multiplier * np.log(n)
    low, high = edge, n - 1 - edge
    fixed_steps = sorted({int(p) for p in fixed_steps if 0 < int(p) < n})
    terms = [(STEP, p) for p in fixed_steps]
    fixed = set(terms)

    _, base_residual = model.basis_and_residual([])
    # Degenerate input (constant series) has nothing to select and would log(0).
    if float(base_residual @ base_residual) <= 1e-12 * max(n, 1):
        return []

    def positions_of(kind, term_list, exclude=None):
        return [p for k, p in term_list if k == kind and (k, p) != exclude]

    # Greedy forward selection with a BIC stop.
    kinds = [KNOT, STEP] if search_steps else [KNOT]
    while len(terms) < max_terms:
        basis, residual = model.basis_and_residual(terms)
        sse = float(residual @ residual)
        if sse <= 1e-12:
            break
        best_gain, best_term = -np.inf, None
        for kind in kinds:
            mask = _spacing_mask(n, positions_of(kind, terms), min_segment, low, high)
            gains = _candidate_gains(model.candidates, basis, residual, kind, mask)
            position = int(np.argmax(gains))
            if gains[position] > best_gain:
                best_gain, best_term = gains[position], (kind, position)
        if best_term is None or not np.isfinite(best_gain) or best_gain >= sse:
            break
        if n * np.log(sse / (sse - best_gain)) < penalty:
            break
        terms.append(best_term)

    # Greedy selection leaves redundant near-duplicate pairs around one true break;
    # coordinate-descent relocation plus backward elimination removes them.
    for _ in range(max_refine_rounds):
        changed = False
        for index in range(len(terms)):
            kind, position = terms[index]
            if (kind, position) in fixed:
                continue
            others = terms[:index] + terms[index + 1 :]
            basis, residual = model.basis_and_residual(others)
            same_kind = np.asarray(positions_of(kind, others), dtype=int)
            left = same_kind[same_kind < position]
            right = same_kind[same_kind > position]
            window_low = max(low, (left.max() + min_segment) if left.size else low)
            window_high = min(high, (right.min() - min_segment) if right.size else high)
            mask = np.zeros(n, dtype=bool)
            if window_high >= window_low:
                mask[window_low : window_high + 1] = True
            gains = _candidate_gains(model.candidates, basis, residual, kind, mask)
            if not np.isfinite(gains).any():
                continue
            best = int(np.argmax(gains))
            if best != position and gains[best] > gains[position] + 1e-12:
                terms[index] = (kind, best)
                changed = True
        if terms:
            _, drop_cost, t_stats, sse, _, inflation = model.term_statistics(
                terms, return_inflation=True
            )
            if not effective_n_prune:
                inflation = 1.0
            prunable = [i for i, term in enumerate(terms) if term not in fixed]
            if prunable:
                weakest = min(prunable, key=lambda i: drop_cost[i])
                # Effective-sample-size BIC: autocorrelated residuals (filtered input,
                # seasonal misfit) make a white-noise BIC keep curvature-chasing
                # knot pairs, so the likelihood gain is divided by the AR(1) inflation.
                log_ratio = np.log((sse + drop_cost[weakest]) / max(sse, 1e-300))
                if n * log_ratio / inflation < penalty:
                    terms.pop(weakest)
                    changed = True
                    continue
            merged = _merge_closest_pair(
                model, terms, fixed, min_segment, low, high, penalty, sse, inflation
            )
            if merged is not None:
                terms = merged
                changed = True
                continue
            step_indices = [i for i, (kind, _) in enumerate(terms) if kind == STEP]
            if step_indices:
                weakest_step = min(step_indices, key=lambda i: abs(t_stats[i]))
                if abs(t_stats[weakest_step]) < step_t_threshold:
                    terms.pop(weakest_step)
                    changed = True
        if not changed:
            break
    return sorted(terms, key=lambda term: (term[1], term[0]))


def joint_fit(values, terms, nuisance=None):
    """One least-squares fit of the selected terms.

    Returns dict with ``trend`` (intercept, slope, hinges), ``level_shift`` (steps),
    per-term ``coefficient`` (hinge: slope change per point; step: shift size) and
    AR(1)-adjusted ``t_stat``, plus ``variance`` of the residual.
    """
    model = JointTrendModel(values, nuisance)
    beta, _, t_stats, _, variance = model.term_statistics(terms)
    design = model.design(terms)
    n_base = model.base.shape[1]
    trend_columns = list(range(model.n_trend_base)) + [
        n_base + i for i, (kind, _) in enumerate(terms) if kind == KNOT
    ]
    step_columns = [n_base + i for i, (kind, _) in enumerate(terms) if kind == STEP]
    trend = design[:, trend_columns] @ beta[trend_columns]
    level_shift = (
        design[:, step_columns] @ beta[step_columns]
        if step_columns
        else np.zeros(model.n)
    )
    details = []
    for i, (kind, position) in enumerate(terms):
        coefficient = float(beta[n_base + i])
        if kind == KNOT:
            # Hinge columns use t scaled by 1/n, so slope change per point is beta / n.
            coefficient = coefficient / model.n
        details.append(
            {
                'kind': kind,
                'position': int(position),
                'coefficient': coefficient,
                't_stat': float(t_stats[i]),
            }
        )
    return {
        'trend': trend,
        'level_shift': level_shift,
        'terms': details,
        'variance': float(variance),
    }


def split_transient_step_pairs(step_details, max_gap, cancel_ratio=0.5):
    """Pair adjacent opposite-sign steps that nearly cancel within ``max_gap`` points.

    A holiday block or multi-week anomaly otherwise enters the trend as two
    permanent level shifts. Returns (kept_steps, transients) where each transient
    is {'start', 'end', 'magnitude'} with magnitude the effect while it lasted.
    """
    ordered = sorted(step_details, key=lambda item: item['position'])
    kept, transients = [], []
    index = 0
    while index < len(ordered):
        current = ordered[index]
        if index + 1 < len(ordered):
            following = ordered[index + 1]
            a, b = current['coefficient'], following['coefficient']
            if (
                following['position'] - current['position'] <= max_gap
                and np.sign(a) != np.sign(b)
                and abs(a + b) < cancel_ratio * max(abs(a), abs(b))
            ):
                transients.append(
                    {
                        'start': current['position'],
                        'end': following['position'],
                        'magnitude': a,
                    }
                )
                index += 2
                continue
        kept.append(current)
        index += 1
    return kept, transients


def profile_likelihood_windows(
    values, terms, nuisance=None, min_segment=28, edge=0, critical_value=3.84
):
    """95% date window per term from the profile likelihood of its position.

    Each term is moved between its same-kind neighbours with all others held;
    positions whose SSE rises by at most ``critical_value * variance_AR`` (chi^2_1)
    over the best position form the window. Returns [(low, high)] aligned to terms.
    """
    model = JointTrendModel(values, nuisance)
    n = model.n
    if not terms or not np.all(np.isfinite(model.values)):
        return [(int(p), int(p)) for _, p in terms]
    _, _, _, _, variance = model.term_statistics(terms)
    variance = max(variance, 1e-300)
    windows = []
    for index, (kind, position) in enumerate(terms):
        others = terms[:index] + terms[index + 1 :]
        basis, residual = model.basis_and_residual(others)
        same_kind = np.array([p for k, p in others if k == kind], dtype=int)
        left = same_kind[same_kind < position]
        right = same_kind[same_kind > position]
        window_low = max(edge, (left.max() + min_segment) if left.size else 1)
        window_high = min(
            n - 1 - edge, (right.min() - min_segment) if right.size else n - 2
        )
        mask = np.zeros(n, dtype=bool)
        if window_high >= window_low:
            mask[window_low : window_high + 1] = True
        gains = _candidate_gains(model.candidates, basis, residual, kind, mask)
        if not np.isfinite(gains).any():
            windows.append((int(position), int(position)))
            continue
        within = np.flatnonzero(
            np.isfinite(gains) & (gains.max() - gains <= critical_value * variance)
        )
        within = np.union1d(within, [position])
        windows.append((int(within.min()), int(within.max())))
    return windows
