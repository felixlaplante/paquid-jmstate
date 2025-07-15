import copy
import warnings
from collections import defaultdict
from typing import Any, DefaultDict, Dict, cast

import torch
from tqdm import tqdm

from ._hazard import HazardMixin
from ._sampler import MetropolisHastingsSampler
from .utils import *


class MultiStateJointModel(HazardMixin):
    """A class of the nonlinear multistate joint model. It feature possibility
    to simulate data, fit based on stochastic gradient with any torch.optim
    optimizer of choice.

    Args:
        HazardMixin (_type_): A mixin class to delegate hazard related methods.
    """

    def __init__(
        self,
        model_design: ModelDesign,
        init_params: ModelParams,
        *,
        n_quad: int = 16,
        n_bissect: int = 16,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing regression, base hazard and link functions and model dimensions.
            n_quad (int, optional): The used numnber of points for Gauss-Legendre quadrature. Defaults to 16.
            n_bissect (int, optional): The number of bissection steps used in transition sampling. Defaults to 16.
        """

        # Store model components
        self.model_design = model_design
        self.params_ = copy.deepcopy(init_params)

        # Set up numerical integration
        self.n_quad = n_quad
        self._std_nodes = None
        self._std_weights = None
        self._legendre_quad(self.n_quad)

        # Set up for bissection algorithm
        self.n_bissect = n_bissect

        # Initialize attributes that will be set during fitting
        self.sampler_: MetropolisHastingsSampler | None = None
        self.fim_: torch.Tensor | None = None
        self.fit_ = False

    def _hazard_ll(self, psi: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the hazard log likelihood.

        Args:
            psi (torch.Tensor): A matrix of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.

        Returns:
            torch.Tensor: The computed log likelihood.
        """

        ll = torch.zeros(data.size)

        for key, bucket in data.buckets_.items():
            alpha, beta = self.params_.alphas[key], self.params_.betas[key]
            idx, t0, t1, obs = bucket

            obs_ll, alts_ll = self._log_and_cum_hazard(
                t0,
                t1,
                data.x[idx],
                psi[idx],
                alpha,
                beta,
                *self.model_design.surv[key],
            )

            # Check for invalid values
            if obs_ll.isnan().any() or obs_ll.isinf().any():
                warnings.warn(f"Invalid observed log likelihood for bucket {key}")
                continue

            if alts_ll.isnan().any() or alts_ll.isinf().any():
                warnings.warn(f"Invalid cumulative hazard for bucket {key}")
                continue

            vals = obs * obs_ll - alts_ll
            ll.scatter_add_(0, idx, vals)

        return ll

    def _long_ll(self, psi: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the longitudinal log likelihood.

        Args:
            psi (torch.Tensor): A matrix of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.

        Returns:
            torch.Tensor: The computed log likelihood.
        """

        # Compute residuals: observed - predicted (only for valid observations)
        predicted = self.model_design.h(data.valid_t_, data.x, psi)
        diff = data.valid_y_ - predicted * data.valid_mask_

        # Check for invalid predictions
        if torch.isnan(predicted).any() or torch.isinf(predicted).any():
            warnings.warn("Invalid predictions encountered in longitudinal model")

        # Reconstruct precision matrix R_inv from Cholesky parametrization
        R_inv = tril_from_flat(self.params_.R_inv, self.params_.R_dim_)

        # Compute log determinant: log det = -2 * sum(log(diag(R_inv)))
        log_det_R = -torch.diag(R_inv).sum() * 2

        # Compute the precision matrix
        R_inv = precision_from_log_cholesky(R_inv)

        # Compute quadratic form: diff.T @ R_inv @ diff for each individual
        quad_form = torch.einsum("ijk,kl,ijl->i", diff, R_inv, diff)

        # Log-likelihood: -0.5 * (log|R| * n_valid + quadratic_form)
        ll = -0.5 * (log_det_R * data.n_valid_ + quad_form)

        # Validate output
        if torch.isnan(ll).any() or torch.isinf(ll).any():
            warnings.warn("Invalid longitudinal likelihood computed")

        return ll

    def _pr_ll(self, b: torch.Tensor) -> torch.Tensor:
        """Computes the prior log likelihood.

        Args:
            b (torch.Tensor): The individual random effects.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            torch.Tensor: The computed log likelihood.
        """

        # Reconstruct precision matrix Q_inv from Cholesky parametrization
        Q_inv = tril_from_flat(self.params_.Q_inv, self.params_.Q_dim_)

        # Compute log determinant: log det = -2 * sum(log(diag(Q_inv)))
        log_det_Q = -torch.diag(Q_inv).sum() * 2

        # Compute the precision matrix
        Q_inv = precision_from_log_cholesky(Q_inv)

        # Compute quadratic form: b.T @ Q_inv @ b for each individual
        quad_form = torch.einsum("ik,kl,il->i", b, Q_inv, b)

        # Log-likelihood: -0.5 * (log det + quadratic_form)
        ll = -0.5 * (log_det_Q + quad_form)

        # Validate output
        if torch.isnan(ll).any() or torch.isinf(ll).any():
            warnings.warn("Invalid prior likelihood computed")

        return ll

    def _ll(self, b: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the total log likelihood up to a constant.

        Args:
            b (torch.Tensor): The individual random effects.
            data (ModelData): Dataset on which the likeihood is computed.

        Returns:
            torch.Tensor: The computed total log likelihood.
        """

        # Transform random effects to individual-specific parameters
        psi = self.model_design.f(self.params_.gamma, b)

        # Validate transformation
        if torch.isnan(psi).any() or torch.isinf(psi).any():
            warnings.warn("Invalid psi values from transformation")

        # Compute individual likelihood components
        long_ll = self._long_ll(psi, data)
        hazard_ll = self._hazard_ll(psi, data)
        prior_ll = self._pr_ll(b)

        # Sum all likelihood components
        total_ll = long_ll + hazard_ll + prior_ll

        # Final validation
        if torch.isnan(total_ll).any() or torch.isinf(total_ll).any():
            warnings.warn("Invalid total likelihood computed")

        return total_ll

    def _build_vec_rep(
        self, trajectories: list[Traj], c: torch.Tensor
    ) -> dict[tuple[int, int], tuple[torch.Tensor, ...]]:
        """Build vectorizable bucket representation.

        Args:
            trajectories (list[Traj]): The trajectories.
            c (torch.Tensor): Censoring times.

        Raises:
            ValueError: If some keys are not in self.surv.
            RuntimeError: If the building fails.

        Returns:
            dict[tuple[int, int], tuple[torch.Tensor, ...]]: The vectorizable buckets representation.
        """

        try:
            # Get survival transitions defined in the model
            trans = set(self.model_design.surv.keys())

            # Build alternative state mapping
            alt_map: DefaultDict[int, list[int]] = defaultdict(list)
            for from_state, to_state in trans:
                alt_map[from_state].append(to_state)

            # Initialize buckets
            buckets: DefaultDict[tuple[int, int], list[list[Any]]] = defaultdict(
                lambda: [[], [], [], []]
            )

            # Process each individual trajectory
            for i, trajectory in enumerate(trajectories):
                # Add censoring
                ext_trajectory = trajectory + [(float(c[i]), None)]

                for (t0, s0), (t1, s1) in zip(ext_trajectory[:-1], ext_trajectory[1:]):
                    if t0 >= t1:
                        continue

                    if s1 is not None and (s0, s1) not in trans:
                        raise ValueError(
                            f"Transition {(s0, s1)} must be in model_design.surv keys"
                        )

                    for alt_state in alt_map.get(s0, []):
                        key = (s0, alt_state)
                        buckets[key][0].append(i)
                        buckets[key][1].append(t0)
                        buckets[key][2].append(t1)
                        buckets[key][3].append(alt_state == s1)

            processed_buckets: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {
                key: (
                    torch.tensor(vals[0], dtype=torch.int64),
                    torch.tensor(vals[1], dtype=torch.float32),
                    torch.tensor(vals[2], dtype=torch.float32),
                    torch.tensor(vals[3], dtype=torch.bool),
                )
                for key, vals in buckets.items()
                if vals[0]
            }

            return processed_buckets

        except Exception as e:
            raise RuntimeError(f"Error building survival buckets: {e}") from e

    def _prepare_data(self, data: ModelData) -> None:
        """Add derived quantities.

        Args:
            data (ModelData): The current dataset.
        """

        # Add derived quantities
        data.valid_mask_ = ~torch.isnan(data.y)
        data.n_valid_ = data.valid_mask_.any(dim=2).sum(dim=1)
        data.valid_t_ = torch.nan_to_num(data.t)
        data.valid_y_ = torch.nan_to_num(data.y)
        data.buckets_ = self._build_vec_rep(data.trajectories, data.c)

    def _setup_mcmc(
        self,
        data: ModelData,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.1,
        target_accept_rate: float = 0.234,
    ) -> MetropolisHastingsSampler:
        """Setup the MCMC kernel and hyperparameters.

        Args:
            data (ModelData): The dataset on which the likelihood is to be computed.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.1.
            target_accept_rate (float, optional): Mean acceptance target. Defaults to 0.234.

        Returns:
            MetropolisHastingsSampler: The intialized Markov kernel.
        """

        # Initialize random effects
        init_b = torch.zeros((data.size, self.params_.Q_dim_))

        # Create sampler
        sampler = MetropolisHastingsSampler(
            log_prob_fn=lambda b: self._ll(b, data),
            init_state=init_b,
            init_step_size=init_step_size,
            adapt_rate=adapt_rate,
            target_accept_rate=target_accept_rate,
        )

        return sampler

    def _mcmc_avg_ll(
        self, sampler: MetropolisHastingsSampler, batch_size: int
    ) -> torch.Tensor:
        """Accumulate the mean log likelihood on a MCMC chain batch.

        Args:
            sampler (MetropolisHastingsSampler): The Markov kernel.
            batch_size (int): The number of steps to accumulate.

        Returns:
            torch.Tensor: The mean log likelihood.
        """

        # Run batch sampling
        total_ll = torch.tensor(0.0)

        for _ in range(batch_size):
            _, curr_log_prob = sampler.step()
            total_ll += curr_log_prob.sum()

        avg_ll = total_ll / batch_size
        return avg_ll

    def fit(
        self,
        data: ModelData,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: Dict[str, Any] = {"lr": 1e-2},
        *,
        n_iter: int = 2000,
        batch_size: int = 1,
        callback: Any | None = None,
        n_iter_fim: int = 1000,
        step_size: float = 0.1,
        adapt_rate: float = 0.1,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
    ) -> None:
        """Fits the MultiStateJointModel.

        Args:
            data (ModelData): The dataset to learn from.
            optimizer (type[torch.optim.Optimizer], optional): The stochastic optimizer constructor. Defaults to torch.optim.Adam.
            optimizer_params (_type_, optional): Optimizer parameter dict. Defaults to {"lr": 1e-2}.
            n_iter (int, optional): Number of iterations for optimization. Defaults to 2000.
            batch_size (int, optional): Batch size used in fitting. Defaults to 1.
            callback (Any | None, optional): A callback function that can be used to track the optimization. Defaults to None.
            n_iter_fim (int, optional): Number of iterations to compute n_iter_fim. Defaults to 1000.
            step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.1.
            target_accept_rate (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
        """

        # Complete data
        self._prepare_data(data)

        # Set up optimizer
        self.params_.require_grad(True)
        params_list = self.params_.as_list
        optimizer_instance = optimizer(params=params_list, **optimizer_params)

        # Set up MCMC
        self.sampler_ = self._setup_mcmc(data, step_size, adapt_rate, accept_target)

        # Warmup MCMC
        self.sampler_.warmup(init_warmup)

        # Main fitting loop
        for iteration in tqdm(range(n_iter), desc="Fitting joint model"):
            try:
                # MCMC: Sample random effects
                self.sampler_.warmup(cont_warmup)
                avg_ll = self._mcmc_avg_ll(self.sampler_, batch_size)

                # Optimization step: Update parameters
                optimizer_instance.zero_grad()
                nll = -avg_ll
                nll.backward()  # type: ignore

                optimizer_instance.step()

                # Execute callback
                if callback is not None:
                    callback()

            except Exception as e:
                warnings.warn(f"Error in iteration {iteration}: {e}")
                continue

        # Set fit_ to True
        self.fit_ = True

        # Compute Fisher Information Matrix
        self._compute_fim(n_iter_fim, cont_warmup)

    def _compute_fim(self, n_iter_fim: int, cont_warmup: int) -> None:
        """Computes the Fisher Information Matrix.

        Args:
            n_iter_fim (int): Number of iterations to calculate Fisher Information Matrix.
            cont_warmup (int): The number of in-between warmup steps.

        Raises:
            ValueError: If self.sampler_ is None.
        """

        if not self.fit_:
            warnings.warn(
                "Model should be fit before computing Fisher Information Matrix"
            )

        if self.sampler_ is None:
            raise ValueError("self.sampler_ must not be None")

        # Setup 
        self.params_.require_grad(True)
        params_list = self.params_.as_list
        d = self.params_.numel
        self.fim_ = torch.zeros(d, d)

        for _ in tqdm(range(n_iter_fim), desc="Computing Fisher Information Matrix"):
            # Sample random effects
            self.sampler_.warmup(cont_warmup)
            _, curr_ll = self.sampler_.step()

            # Clear gradients
            for p in params_list:
                if p.grad is not None:
                    p.grad.zero_()

            # Compute gradients
            ll = curr_ll.sum()
            ll.backward()  # type: ignore

            # Collect gradient vector
            grad_chunks: list[torch.Tensor] = []
            for p in params_list:
                if p.grad is not None:
                    grad_chunks.append(p.grad.view(-1))
                else:
                    grad_chunks.append(torch.zeros(p.numel()))

            grad = torch.cat(grad_chunks)

            # Update Fisher Information Matrix
            self.fim_ += torch.outer(grad, grad) / n_iter_fim

        if torch.isnan(self.fim_).any() or torch.isinf(self.fim_).any():
            warnings.warn("Error computing Fisher Information Matrix")
            self.fim_ = None

    def get_stderror(self) -> ModelParams:
        """Returns the standard error of the parameters that can be used to
        draw confidence intervals.

        Raises:
            ValueError: If the Fisher Information Matrix could not be computed.

        Returns:
            ModelParams: The standard error in the same format as the parameters.
        """

        # Check if self.fim_ is well defined
        if self.fim_ is None:
            raise ValueError(
                "Fisher Information Matrix inference failed. CIs may not be computed."
            )

        # Get parameter vector
        params_list = self.params_.as_list
        params_flat = torch.cat([p.detach().flatten() for p in params_list])

        # Compute standard errors
        try:
            fim_inv = torch.linalg.pinv(self.fim_)  # type: ignore
            flat_se = torch.sqrt(fim_inv.diag())  # type: ignore

        except Exception as e:
            warnings.warn(f"Error inverting Fisher Information Matrix: {e}")
            flat_se = torch.full_like(params_flat, torch.nan)

        # Organize by parameter structure
        se: dict[str, Any] = {}
        i = 0

        for key in ["gamma", "Q_inv", "R_inv", "alphas", "betas"]:
            val = getattr(self.params_, key)
            if isinstance(val, dict):
                param_dict = {}
                for subkey, subval in cast(
                    list[tuple[tuple[int, int], torch.Tensor]], val.items()
                ):
                    n = subval.numel()
                    shape = subval.shape
                    param_dict[subkey] = flat_se[i : i + n].view(shape)
                    i += n
                se[key] = param_dict  # assign entire dict to field
            else:
                n = val.numel()
                shape = val.shape
                se[key] = flat_se[i : i + n].view(shape)  # assign tensor
                i += n

        return ModelParams(**se)

    def sample_trajectories(
        self,
        sample_data: SampleData,
        c_max: torch.Tensor,
        max_length: int = 100,
    ) -> list[Traj]:
        """Sample future trajectories from the fitted joint model.

        Args:
            sample_data (SampleData): Prediction data.
            c_max (torch.Tensor): The maximum trajectory sampling time (censoring time).
            max_length (int, optional): Maximum iterations or sampling (prevents infinite loops). Defaults to 100.

        Raises:
            ValueError: If all the parameters are not set.
            ValueError: If the shape of c_max is not compatible.
            RuntimeError: If the sampling fails.

        Returns:
            list[Traj]: The sampled trajectories.
        """

        try:
            # Convert and check if c_max matches the right shape
            c_max = torch.as_tensor(c_max, dtype=torch.float32)
            if c_max.shape != (sample_data.size,):
                raise ValueError(
                    "c_max has incorrect shape, got {c_max.shape}, expected {(sample_data.size,)}"
                )

            # Initialize with copies of current trajectories
            trajectories = [list(trajectory) for trajectory in sample_data.trajectories]

            # Get initial buckets from last states
            last_states = [trajectory[-1:] for trajectory in trajectories]
            current_buckets = self._build_vec_rep(last_states, c_max)

            # Sample future transitions iteratively
            for iteration in range(max_length):
                # Stop if no more possible transitions
                if not current_buckets:
                    break

                # Initialize candidate transition times
                n_transitions = len(current_buckets)
                t_candidates = torch.full(
                    (sample_data.size, n_transitions), torch.inf, dtype=torch.float32
                )

                # Sample transition times for each possible transition
                for j, (transition_key, bucket_info) in enumerate(
                    current_buckets.items()
                ):
                    try:
                        # Get parameters for this transition
                        alpha = self.params_.alphas[transition_key]
                        beta = self.params_.betas[transition_key]

                        # Extract bucket information
                        idx, t0, t1, _ = bucket_info

                        # Sample transition times
                        t_sample = self._sample_trajectory_step(
                            t0,
                            t1 + 1e-8,  # Extend upper bound
                            sample_data.x[idx],
                            sample_data.psi[idx],
                            alpha,
                            beta,
                            *self.model_design.surv[transition_key],
                            c=(
                                sample_data.c[idx]
                                if not iteration and sample_data.c is not None
                                else None
                            ),
                            n_bissect=self.n_bissect,
                        )

                        # Store candidate times
                        t_candidates[idx, j] = t_sample

                    except Exception as e:
                        warnings.warn(
                            f"Error sampling transition {transition_key}: {e}"
                        )
                        continue

                # Find earliest transition
                min_times, argmin_indices = torch.min(t_candidates, dim=1)

                # Identify indivuals with valid transitions
                valid_indivs = torch.nonzero(torch.isfinite(min_times)).flatten()

                # Update trajectories with new transitions
                for indiv_idx in valid_indivs:
                    indiv_idx = int(indiv_idx)
                    transition_idx = int(argmin_indices[indiv_idx])
                    transition_time = float(min_times[indiv_idx])

                    # Get the new state from the transition
                    transition_key = list(current_buckets.keys())[transition_idx]
                    new_state = transition_key[1]  # to_state

                    # Add transition to trajectory
                    trajectories[indiv_idx].append((transition_time, new_state))

                # Update buckets for next iteration
                last_states = [trajectory[-1:] for trajectory in trajectories]
                current_buckets = self._build_vec_rep(last_states, c_max)

            # Remove transitions that exceed censoring times
            for i, trajectory in enumerate(trajectories):
                censoring_time = float(c_max[i])
                # If trajectory was truncated, ensure it doesn't end beyond censoring
                if trajectory[-1][0] > censoring_time:
                    trajectories[i] = trajectory[:-1]

            return trajectories

        except Exception as e:
            raise RuntimeError(f"Error in trajectory sampling: {e}") from e

    def predict_trajectories(
        self,
        pred_data: ModelData,
        c_max: torch.Tensor,
        n_iter_b: int,
        n_iter_T: int,
        step_size: float = 0.1,
        adapt_rate: float = 0.1,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        max_length: int = 100,
    ) -> list[list[list[Traj]]]:
        """Predict survival trajectories for new individuals.

        Args:
            pred_data (ModelData): Prediction data.
            c_max (torch.Tensor): Maximum prediction times.
            n_iter_b (int): Number of iterations for random effects sampling.
            n_iter_T (int): Number of trajectory samples per random effects sample.
            step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.1.
            target_accept_rate (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            max_length (int, optional): Maximum iterations or sampling (prevents infinite loops). Defaults to 100.

        Raises:
            RuntimeError: If the prediction fails.

        Returns:
            list[list[list[Traj]]]: A list of lists of trajectories. First list is for a b sample, then multiples iid drawings of the trajectories.
        """

        try:
            # Convert and check if c_max matches the right shape
            c_max = torch.as_tensor(c_max, dtype=torch.float32)
            if c_max.shape != (pred_data.size,):
                raise ValueError(
                    "c_max has incorrect shape, got {c_max.shape}, expected {(sample_data.size,)}"
                )

            # Load and validate prediction data
            self._prepare_data(pred_data)

            # Set up MCMC for prediction
            sampler = self._setup_mcmc(pred_data, step_size, adapt_rate, accept_target)

            # Warmup MCMC
            sampler.warmup(init_warmup)

            # Prepare replicate data for trajectory sampling
            x_rep = pred_data.x.repeat(n_iter_T, 1)
            trajectories_rep = pred_data.trajectories * n_iter_T
            c_rep = pred_data.c.repeat(n_iter_T)
            c_max_rep = c_max.repeat(n_iter_T)

            # Generate predictions
            predictions: list[list[list[Traj]]] = []

            for _ in tqdm(range(n_iter_b), desc="Predicting trajectories"):
                # Sample random effects
                sampler.warmup(cont_warmup)

                curr_b, _ = sampler.step()

                # Transform to individual-specific parameters
                psi = self.model_design.f(self.params_.gamma, curr_b)

                # Replicate for multiple trajectory samples
                psi_rep = psi.repeat(n_iter_T, 1)

                sample_data = SampleData(x_rep, trajectories_rep, psi_rep, c_rep)

                # Sample trajectories
                trajectories = self.sample_trajectories(
                    sample_data, c_max_rep, max_length
                )

                # Organize by trajectory iteration
                trajectory_chunks = [
                    trajectories[i * pred_data.size : (i + 1) * pred_data.size]
                    for i in range(n_iter_T)
                ]

                predictions.append(trajectory_chunks)

            return predictions

        except Exception as e:
            raise RuntimeError(f"Error in survival prediction: {e}") from e
