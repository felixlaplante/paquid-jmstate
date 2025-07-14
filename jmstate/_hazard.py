import warnings
from typing import Any, cast

import numpy as np
import torch

from .utils import *


class HazardMixin:
    """Mixin class for hazard model computations."""

    def _legendre_quad(self, n_quad: int = 16) -> None:
        """Get the Legendre quadrature nodes and weights.

        Args:
            n_quad (int, optional): The number of quadrature points. Defaults to 16.
        """
        nodes, weights = cast(
            tuple[
                np.ndarray[Any, np.dtype[np.float64]],
                np.ndarray[Any, np.dtype[np.float64]],
            ],
            np.polynomial.legendre.leggauss(n_quad),  #  type: ignore
        )
        self._std_nodes = torch.tensor(nodes, dtype=torch.float32)
        self._std_weights = torch.tensor(weights, dtype=torch.float32)

    def _log_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_lambda0: BaseFun,
        g: LinkFun,
    ) -> torch.Tensor:
        """Computes log hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_lambda0 (BaseFun): Base hazard function.
            g (LinkFun): Link function.

        Returns:
            torch.Tensor: The computed log hazard.
        """

        # Compute baseline hazard
        base = log_lambda0(t1, t0)

        # Compute time-varying effects
        mod = g(t1, x, psi)

        # Validate g output
        if torch.isnan(mod).any() or torch.isinf(mod).any():
            warnings.warn("Invalid values in g(t,x,psi)")

        # Compute log hazard
        log_hazard = (
            base + torch.einsum("ijk,k->ij", mod, alpha) + x @ beta.unsqueeze(1)
        )

        return log_hazard

    def _cum_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_lambda0: BaseFun,
        g: LinkFun,
    ) -> torch.Tensor:
        """Computes cumulative hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_lambda0 (BaseFun): Base hazard function.
            g (LinkFun): Link function.

        Returns:
            torch.Tensor: The computed cumulative hazard.
        """

        # Reshape for broadcasting
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)

        # Transform to quadrature interval [-1, 1]
        mid = 0.5 * (t0 + t1)
        half = 0.5 * (t1 - t0)

        # Evaluate at quadrature points
        ts = mid + half * self._std_nodes

        # Compute hazard at quadrature points
        log_hazard_vals = self._log_hazard(t0, ts, x, psi, alpha, beta, log_lambda0, g)

        # Numerical integration using Gaussian quadrature
        hazard_vals = torch.exp(torch.clamp(log_hazard_vals, min=-50.0, max=50.0))

        # Check for numerical issues
        if torch.isnan(hazard_vals).any() or torch.isinf(hazard_vals).any():
            warnings.warn("Numerical issues in hazard computation")
            hazard_vals = torch.nan_to_num(hazard_vals, nan=0.0, posinf=1e10)

        integral = half.flatten() * (hazard_vals * self._std_weights).sum(dim=1)

        return integral

    def _log_and_cum_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_lambda0: BaseFun,
        g: LinkFun,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes both log and cumulative hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_lambda0 (BaseFun): Base hazard function.
            g (LinkFun): Link function.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing log and cumulative hazard.
        """

        # Reshape for broadcasting
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)

        # Transform to quadrature interval
        mid = 0.5 * (t0 + t1)
        half = 0.5 * (t1 - t0)

        # Combine endpoint and quadrature points
        ts = torch.cat([t1, mid + half * self._std_nodes], dim=1)

        # Compute log hazard at all points
        temp = self._log_hazard(t0, ts, x, psi, alpha, beta, log_lambda0, g)

        # Extract log hazard at endpoint and quadrature points
        log_hazard = temp[:, :1]  # Log hazard at t1
        quad_vals = torch.exp(
            torch.clamp(temp[:, 1:], min=-50.0, max=50.0)
        )  # Hazard at quadrature points

        # Compute cumulative hazard using quadrature
        integral = half.flatten() * (quad_vals * self._std_weights).sum(dim=1)

        return log_hazard.flatten(), integral

    def _sample_trajectory_step(
        self,
        t_left: torch.Tensor,
        t_right: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_lambda0: BaseFun,
        g: LinkFun,
        *,
        c: torch.Tensor | None = None,
        n_bissect: int,
    ) -> torch.Tensor:
        """Sample survival times using inverse transform sampling.

        Args:
            t_left (torch.Tensor): Left sampling time.
            t_right (torch.Tensor): Right censoring sampling time.
            x (torch.Tensor): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_lambda0 (BaseFun): Base hazard function.
            g (LinkFun): Link function.
            n_bissect (int): _description_
            c (torch.Tensor | None, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: The computed pre transition times.
        """

        n = x.shape[0]

        # Initialize for bisection search
        t0 = t_left.clone().view(-1, 1)
        t_left, t_right = t_left.view(-1, 1), t_right.view(-1, 1)

        # Generate exponential random variables
        target = -torch.log(torch.clamp(torch.rand(n), min=1e-8))

        # Adjust target if conditioning on existing survival
        if c is not None:
            c = c.view(-1, 1)
            existing_hazard = self._cum_hazard(
                t0, c, x, psi, alpha, beta, log_lambda0, g
            )
            target += existing_hazard

        # Bisection search for survival times
        for _ in range(n_bissect):
            t_mid = 0.5 * (t_left + t_right)

            cumulative = self._cum_hazard(
                t0, t_mid, x, psi, alpha, beta, log_lambda0, g
            )

            # Update search bounds
            accept_mask = cumulative < target
            t_left[accept_mask] = t_mid[accept_mask]
            t_right[~accept_mask] = t_mid[~accept_mask]

        return t_right.flatten()
