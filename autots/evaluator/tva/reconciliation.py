# -*- coding: utf-8 -*-
"""
ReconciliationBridge — Bridge to AutoTS hierarchical reconciliation.

Thin adapter between TVA forecasts and the reconciliation functions in
autots/tools/hierarchial.py (mint_reconcile, erm_reconcile, etc.).
"""

import numpy as np
import pandas as pd

from autots.tools.hierarchial import (
    mint_reconcile,
    erm_reconcile,
    ledoit_wolf_covariance,
    volatility_weighted_mint_reconcile,
    iterative_mint_reconcile,
)


class ReconciliationBridge:
    """Bridges TVA forecasts to hierarchical reconciliation methods.

    Handles S matrix construction, covariance estimation, and method dispatch.

    Args:
        method: Reconciliation method ('mint', 'erm', 'iterative_mint',
            'volatility_mint'). Default 'mint'.
        covariance_method: How to estimate W ('ledoit_wolf', 'identity', 'sample').
    """

    def __init__(
        self,
        method: str = 'mint',
        covariance_method: str = 'ledoit_wolf',
    ):
        self.method = method
        self.covariance_method = covariance_method

    def _estimate_covariance(self, residuals: np.ndarray) -> np.ndarray:
        """Estimate covariance matrix W from forecast residuals.

        Args:
            residuals: (T, L) residual matrix for all nodes.

        Returns:
            (L, L) covariance matrix.
        """
        L = residuals.shape[1]
        if self.covariance_method == 'ledoit_wolf':
            return ledoit_wolf_covariance(residuals)
        elif self.covariance_method == 'identity':
            return np.eye(L, dtype=np.float64)
        else:
            # sample covariance with regularization
            cov = np.cov(residuals, rowvar=False)
            cov += np.eye(L) * 1e-6
            return cov

    def reconcile(
        self,
        forecasts: pd.DataFrame,
        S: np.ndarray,
        residuals: np.ndarray = None,
    ) -> pd.DataFrame:
        """Apply hierarchical reconciliation to forecast DataFrame.

        Args:
            forecasts: (T, L) DataFrame with all hierarchy levels.
                Columns should be ordered: aggregate nodes first, then bottom nodes,
                matching the row order of S.
            S: (L, M) summing matrix from YggdrasilPriors.build_hierarchy_matrix().
            residuals: (T_hist, L) historical residuals for covariance estimation.
                If None, uses identity covariance.

        Returns:
            Reconciled DataFrame with same shape and columns as input.
        """
        y_all = forecasts.values  # (T, L)
        L = y_all.shape[1]

        if residuals is not None:
            W = self._estimate_covariance(residuals)
        else:
            W = np.eye(L, dtype=np.float64)

        if self.method == 'mint':
            reconciled = mint_reconcile(S, y_all, W)
        elif self.method == 'erm':
            reconciled = erm_reconcile(S, y_all, W)
        elif self.method == 'iterative_mint':
            reconciled = iterative_mint_reconcile(S, y_all, W)
        elif self.method == 'volatility_mint':
            return self.reconcile_with_volatility(
                forecasts, S, residuals if residuals is not None else np.zeros((1, L))
            )
        else:
            reconciled = mint_reconcile(S, y_all, W)

        return pd.DataFrame(
            reconciled, index=forecasts.index, columns=forecasts.columns
        )

    def reconcile_with_volatility(
        self,
        forecasts: pd.DataFrame,
        S: np.ndarray,
        residuals: np.ndarray,
        cov_bottom: np.ndarray = None,
        volatility_method: str = 'variance',
        volatility_power: float = 1.0,
        alpha: float = 0.5,
    ) -> pd.DataFrame:
        """Volatility-weighted reconciliation for covariance-aware adjustment.

        Args:
            forecasts: (T, L) DataFrame.
            S: (L, M) summing matrix.
            residuals: (T_hist, L) historical residuals.
            cov_bottom: (M, M) bottom-level covariance. Estimated if None.
            volatility_method: 'variance', 'std', or 'cv'.
            volatility_power: Power to raise volatility weights to.
            alpha: Blend between standard W and volatility weights.

        Returns:
            Reconciled DataFrame.
        """
        y_all = forecasts.values
        W = self._estimate_covariance(residuals)

        M = S.shape[1]
        if cov_bottom is None:
            # extract bottom-level residuals (last M columns)
            bottom_residuals = residuals[:, -M:]
            cov_bottom = self._estimate_covariance(bottom_residuals)

        reconciled = volatility_weighted_mint_reconcile(
            S,
            y_all,
            W,
            cov_bottom,
            volatility_method=volatility_method,
            volatility_power=volatility_power,
            alpha=alpha,
        )
        return pd.DataFrame(
            reconciled, index=forecasts.index, columns=forecasts.columns
        )
