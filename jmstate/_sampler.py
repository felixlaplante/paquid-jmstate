import warnings
from typing import Callable

import torch


class MetropolisHastingsSampler:
    """A robust Metropolis-Hastings sampler with adaptive step size."""

    def __init__(
        self,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        init_state: torch.Tensor,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.1,
        target_accept_rate: float = 0.234,
    ):
        """Initialize the Metropolis-Hastings sampler kernel.

        Args:
            log_prob_fn (Callable[[torch.Tensor], torch.Tensor]): Function that computes log probability.
            init_state (torch.Tensor): Starting state for the chain.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.1.
            target_accept_rate (float, optional): Mean acceptance target. Defaults to 0.234.

        Raises:
            RuntimeError: If the initial log prob fails to be computed.
        """

        self.log_prob_fn = log_prob_fn
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate

        # Initialize state
        self.current_state_ = init_state.clone().detach()
        self.step_size_ = torch.tensor(init_step_size)

        # Compute initial log probability
        try:
            self.current_log_prob_ = self.log_prob_fn(self.current_state_)
        except Exception as e:
            raise RuntimeError(f"Failed to compute initial log probability: {e}")

        # Statistics tracking
        self.n_samples = 0
        self.n_accepted = 0

        self._check()

    def _check(self):
        """Check if every input is valid.

        Raises:
            TypeError: If the function is not callable.
            ValueError: If step_size is not strictly positive.
            ValueError: If target_accept_rate is not in (0, 1).
            ValueError: If adapt_rate is not strictly positive.
        """

        if not callable(self.log_prob_fn):
            raise TypeError("log_prob_fn must be callable")

        if self.step_size_ <= 0:
            raise ValueError("step_size must be strictly positive")

        if not 0 < self.target_accept_rate < 1:
            raise ValueError("target_accept_rate must be between 0 and 1")

        if self.adapt_rate <= 0:
            raise ValueError("adapt_rate must be strictly positive")

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a single kernel step.

        Raises:
            RuntimeError: If the step failed.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing current_state and current_log_prob.
        """

        try:
            # Detach current state to avoid gradient accumulation
            self.current_state_ = self.current_state_.detach()
            self.current_log_prob_ = self.current_log_prob_.detach()

            # Generate proposal
            noise = torch.randn_like(self.current_state_)
            proposed_state = self.current_state_ + noise * self.step_size_

            # Compute proposal log probability
            try:
                proposed_log_prob = self.log_prob_fn(proposed_state)
            except Exception as e:
                warnings.warn(f"Failed to compute proposal log probability: {e}")
                return self.current_state_, self.current_log_prob_

            # Check for invalid log probabilities
            if (
                torch.isnan(proposed_log_prob).any()
                or torch.isinf(proposed_log_prob).any()
            ):
                warnings.warn("Invalid log probability encountered in proposal")
                return self.current_state_, self.current_log_prob_

            # Compute acceptance probability
            log_prob_diff = proposed_log_prob - self.current_log_prob_

            # Vectorized acceptance decision
            log_uniform = torch.log(
                torch.clamp(torch.rand_like(log_prob_diff), min=1e-8)
            )
            accept_mask = log_uniform < log_prob_diff

            # Update accepted states
            if accept_mask.any():
                self.current_state_ = torch.where(
                    (
                        accept_mask.unsqueeze(-1)
                        if accept_mask.dim() < self.current_state_.dim()
                        else accept_mask
                    ),
                    proposed_state,
                    self.current_state_,
                )
                self.current_log_prob_ = torch.where(
                    accept_mask, proposed_log_prob, self.current_log_prob_
                )

            # Update statistics
            self.n_samples += 1
            accepted = accept_mask.float().mean().item()
            self.n_accepted += accepted

            # Adapt step size
            self._adapt_step_size(accepted)

            return self.current_state_, self.current_log_prob_

        except Exception as e:
            raise RuntimeError("Kernel step failed: {e}") from e

    def warmup(self, warmup: int) -> None:
        """Warmups the MCMC.

        Args:
            warmup (int): The number of warmup steps.

        Raises:
            ValueError: If the warmup steps is not positive.
            RuntimeError: If the warmup fails.
        """

        try:
            if warmup < 0:
                raise ValueError("Warmup must be a non-negative integer")

            with torch.no_grad():
                for _ in range(warmup):
                    self.step()

        except Exception as e:
            raise RuntimeError("Warmup failed: {e}") from e

    def _adapt_step_size(self, accept_rate: float):
        """Adapt the step_size.

        Args:
            accept_rate (float): The adaptation rate.
        """

        adaptation = (
            torch.tensor(accept_rate - self.target_accept_rate) * self.adapt_rate
        )
        self.step_size_ *= torch.exp(adaptation)

    @property
    def acceptance_rate(self) -> float:
        """Gets the acceptance_rate mean.

        Returns:
            float: The mean of the acceptance_rate accross iterations.
        """

        return self.n_accepted / max(self.n_samples, 1)

    @property
    def current_step_size(self) -> float:
        """Gets current step_size.

        Returns:
            float: The current step_size.
        """

        return self.step_size_.item()
