from math import isqrt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, TypeAlias, cast

import torch


# Aliases
RegFun: TypeAlias = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
LinkFun: TypeAlias = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
EffFun: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BaseFun: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Traj: TypeAlias = list[tuple[float, Any]]


@dataclass
class ModelDesign:
    """Class containing all multistate joint model design.

    Raises:
        TypeError: If f is not callable.
        TypeError: If h is not callable.
        TypeError: If any of the base hazard functions is not callable.
        TypeError: If any of the link functions is not callable.
        ValueError: If the keys of alpha_dims and surv do not match.
    """

    f: EffFun
    h: RegFun
    surv: dict[
        tuple[int, int],
        tuple[
            BaseFun,
            LinkFun,
        ],
    ]

    def __post_init__(self):
        """Runs the post init checks."""
        self._check()

    def _check(self):
        """Runs the checks themselves.

        Raises:
            TypeError: If f is not callable.
            TypeError: If h is not callable.
            TypeError: If any of the base hazard functions is not callable.
            TypeError: If any of the link functions is not callable.
            ValueError: If the keys of alpha_dims and surv do not match.
        """
        if not callable(self.f):
            raise TypeError("f must be callable")
        if not callable(self.h):
            raise TypeError("h must be callable")

        for key, (base_fn, link_fn) in self.surv.items():
            if not callable(base_fn):
                raise TypeError(f"Base hazard function for key {key} must be callable")
            if not callable(link_fn):
                raise TypeError(f"Link function for key {key} must be callable")


@dataclass
class ModelData:
    """Class containing all multistate joint model data.

    Raises:
        ValueError: If the dimensions do not match theoretical dimensions.
        ValueError: If the number of individuals is not consistent.
        ValueError: If t is not broadcastable with y.
        ValueError: If t contains NaNs where y is not.
        ValueError: If trajectories are not sorted.

    Returns:
        _type_: The instance.
    """

    x: torch.Tensor
    t: torch.Tensor
    y: torch.Tensor
    trajectories: list[Traj]
    c: torch.Tensor
    valid_t_: torch.Tensor = field(init=False, repr=False)
    valid_y_: torch.Tensor = field(init=False, repr=False)
    valid_mask_: torch.Tensor = field(init=False, repr=False)
    n_valid_: torch.Tensor = field(init=False, repr=False)
    buckets_: dict[tuple[int, int], tuple[torch.Tensor, ...]] = field(
        init=False, repr=False
    )

    def __post_init__(self):
        """Runs the post init conversions and checks."""

        # Convert to float32
        self.x = torch.as_tensor(self.x, dtype=torch.float32)
        self.t = torch.as_tensor(self.t, dtype=torch.float32)
        self.y = torch.as_tensor(self.y, dtype=torch.float32)
        self.c = torch.as_tensor(self.c, dtype=torch.float32)

        self._check()

    def _check(self):
        """Runs the checks themselves.

        Raises:
            ValueError: If the dimensions do not match theoretical dimensions.
            ValueError: If the number of individuals is not consistent.
            ValueError: If t is not broadcastable with y.
            ValueError: If t contains NaNs where y is not.
            ValueError: If trajectories are not sorted.
        """

        # Validate tensor dimensions
        dim_checks = [
            (self.c, 1, "c"),
            (self.x, 2, "x"),
            (self.y, 3, "y"),
        ]

        for tensor, expected_dim, name in dim_checks:
            if tensor.ndim != expected_dim:
                raise ValueError(
                    f"{name} must be {expected_dim}-dimensional, got {tensor.ndim}"
                )

        # Validate consistent number of individuals
        n = self.size
        if not (
            self.y.shape[0] == n and len(self.trajectories) == n and self.c.numel() == n
        ):
            raise ValueError("Inconsistent number of individuals")

        # Validate time dimension compatibility
        valid_t_shapes = [
            (self.y.shape[1],),
            (1, self.y.shape[1]),
            self.y.shape[:2],
        ]

        if self.t.shape not in valid_t_shapes:
            raise ValueError("t must be broadcastable with y")

        if (
            self.t.shape == self.y.shape[:2]
            and (~self.y.isnan().any(dim=2) & self.t.isnan()).any()
        ):
            raise ValueError("t must not be torch.nan where y is not")
        if not all(
            all(t0 <= t1 for (t0, _), (t1, _) in zip(trajectory[:-1], trajectory[1:]))
            for trajectory in self.trajectories
        ):
            raise ValueError(f"All trajectories bust be sorted in their first argument")

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return self.x.shape[0]


@dataclass
class SampleData:
    """Class containing all multistate joint model sampling data.

    Raises:
        ValueError: If the dimensions do not match theoretical dimensions.
        ValueError: If the number of individuals is not consistent.
        ValueError: If the number of individuals is not consistent for t_surv.
        ValueError: If trajectories are not sorted.

    Returns:
        _type_: _description_
    """

    x: torch.Tensor
    trajectories: list[Traj]
    psi: torch.Tensor
    c: torch.Tensor | None = None

    def __post_init__(self):
        """Runs the post init conversions and checks."""

        # Convert to float32
        self.x = torch.as_tensor(self.x, dtype=torch.float32)
        self.c = (
            torch.as_tensor(self.c, dtype=torch.float32) if self.c is not None else None
        )
        self.psi = torch.as_tensor(self.psi, dtype=torch.float32)

        self._check()

    def _check(self):
        """Runs the post init checks themselves.

        Raises:
            ValueError: If the dimensions do not match theoretical dimensions.
            ValueError: If the number of individuals is not consistent.
            ValueError: If the number of individuals is not consistent for t_surv.
            ValueError: If trajectories are not sorted.
        """

        # Validate tensor dimensions
        dim_checks = [
            (self.c, 1, "c"),
            (self.x, 2, "x"),
            (self.psi, 2, "psi"),
        ]

        for tensor, expected_dim, name in dim_checks:
            if tensor is not None and tensor.ndim != expected_dim:
                raise ValueError(
                    f"{name} must be {expected_dim}-dimensional, got {tensor.ndim}"
                )

        # Validate consistent number of individuals
        n = self.size
        if not (
            self.psi.shape[0] == n
            and len(self.trajectories) == n
            and (self.c is None or self.c.numel() == n)
        ):
            raise ValueError("Inconsistent number of individuals")
        if not all(
            all(t0 <= t1 for (t0, _), (t1, _) in zip(trajectory[:-1], trajectory[1:]))
            for trajectory in self.trajectories
        ):
            raise ValueError(f"All trajectories bust be sorted in their first argument")

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return self.x.shape[0]


@dataclass
class ModelParams:
    """A class containing model optimizable parameters.

    Raises:
        AttributeError: If not all parameters are not set.

    Returns:
        _type_: The instance.
    """

    gamma: torch.Tensor
    Q_info: tuple[torch.Tensor, str]
    R_info: tuple[torch.Tensor, str]
    alphas: dict[tuple[int, int], torch.Tensor]
    betas: dict[tuple[int, int], torch.Tensor]
    Q_flat_: torch.Tensor = field(init=False, repr=False)
    R_flat_: torch.Tensor = field(init=False, repr=False)
    Q_method_: str = field(init=False, repr=False)
    R_method_: str = field(init=False, repr=False)
    Q_dim_: int = field(init=False, repr=False)
    R_dim_: int = field(init=False, repr=False)

    def __post_init__(self):
        """Convert and init to float32 the parameters."""

        # Init
        self.Q_flat_, self.Q_method_ = self.Q_info
        self.R_flat_, self.R_method_ = self.R_info

        # Convert to float32
        self.gamma = torch.as_tensor(self.gamma, dtype=torch.float32)
        self.Q_flat_ = torch.as_tensor(self.Q_flat_, dtype=torch.float32)
        self.R_flat_ = torch.as_tensor(self.R_flat_, dtype=torch.float32)

        # Make pointers
        self.Q_info = (self.Q_flat_, self.Q_method_)
        self.R_info = (self.Q_flat_, self.Q_method_)

        for alpha in self.alphas.values():
            alpha = torch.as_tensor(alpha, dtype=torch.float32)

        for beta in self.betas.values():
            beta = torch.as_tensor(beta, dtype=torch.float32)

        self._check()
        self._set_dims("Q")
        self._set_dims("R")

    def _check(self):
        """Runs the post init checks themselves.

        Raises:
            ValueError: If Q_method_ is not in ("full", "diag", "ball").
            ValueError: If R_method_ is not in ("full", "diag", "ball").
            ValueError: If either gamma, Q_flat_ or R_flat_ is not flat.
            ValueError: If the alphas are not flat.
            ValueError: If the betas are not flat.
        """

        dim_checks = [
            (self.gamma, 1, "gamma"),
            (self.Q_flat_, 1, "Q_flat_"),
            (self.R_flat_, 1, "R_flat_"),
        ]

        # Validate tensors
        for tensor, expected_dim, name in dim_checks:
            if tensor.ndim != expected_dim:
                raise ValueError(
                    f"{name} must be {expected_dim}-dimensional, got {tensor.ndim}"
                )

        # Check dictionary tensors
        for key, alpha in self.alphas.items():
            if alpha.ndim != 1:
                raise ValueError(
                    f"alphas[{key}] must be 1-dimensional, got {alpha.ndim}"
                )

        for key, beta in self.betas.items():
            if beta.ndim != 1:
                raise ValueError(f"betas[{key}] must be 1-dimensional, got {beta.ndim}")

    def _set_dims(self, matrix: str) -> None:
        """Sets dimensions for matrix.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the name matrix is not "Q" nor "R".
            ValueError: If the number of elements is not a triangular number and the method is "full".
            ValueError: If the number of elements is not one and the method is "ball".
        """

        if not matrix in ("Q", "R"):
            raise ValueError(f"matrix should be either Q or R, got {matrix}")

        flat = getattr(self, matrix + "_flat_")
        method = getattr(self, matrix + "_method_")

        match method:
            case "full":
                n = (isqrt(1 + 8 * flat.numel()) - 1) // 2
                if (n * (n + 1)) // 2 != flat.numel():
                    raise ValueError(
                        f"{flat.numel()} is not a triangular number for matrix {matrix}"
                    )
                setattr(self, matrix + "_dim_", n)
            case "diag":
                n = flat.numel()
                setattr(self, matrix + "_dim_", n)
            case "ball":
                if 1 != flat.numel():
                    f"Inocrrect number of elements for flat, got {flat.numel()} but expected {1}"
                setattr(self, matrix + "_dim_", 1)
            case _:
                raise ValueError(f"Got method {method} unknown for matrix {matrix}")

    @property
    def as_list(self) -> list[torch.Tensor]:
        """Get a list of all the parameters for optimization.

        Returns:
            list[torch.Tensor]: The list of the parameters.
        """

        params_list: list[torch.Tensor] = []

        # Add non-dictionary parameters
        params_list.append(self.gamma)
        params_list.append(self.Q_flat_)
        params_list.append(self.R_flat_)

        # Add dictionary parameters
        params_list.extend(self.alphas.values())
        params_list.extend(self.betas.values())

        return params_list

    @property
    def numel(self) -> int:
        """Return the number of parameters.

        Returns:
            int: The number of the parameters.
        """

        return sum([p.numel() for p in self.as_list])

    def get_precision(self, matrix: str) -> torch.Tensor:
        """Get precision matrix.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the matrix is not in ("Q", "R")

        Returns:
            torch.Tensor: The precision matrix.
        """

        if not matrix in ("Q", "R"):
            raise ValueError(f"matrix should be either Q or R, got {matrix}")

        # Get flat then log cholesky
        flat = getattr(self, matrix + "_flat_")
        n = getattr(self, matrix + "_dim_")
        method = getattr(self, matrix + "_method_")

        L = log_cholesky_from_flat(flat, n, method)
        P = precision_from_log_cholesky(L)

        return P

    def get_precision_and_logdet(
        self, matrix: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get precision matrix as well as log determinant.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the matrix is not in ("Q", "R")

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The tuple of precision matrix and log determinannt of precision.
        """

        if not matrix in ("Q", "R"):
            raise ValueError(f"matrix should be either Q or R, got {matrix}")

        # Get flat then log cholesky
        flat = getattr(self, matrix + "_flat_")
        n = getattr(self, matrix + "_dim_")
        method = getattr(self, matrix + "_method_")

        L = log_cholesky_from_flat(flat, n, method)
        logdet = 2 * L.diag().sum()
        P = precision_from_log_cholesky(L)

        return P, logdet

    def require_grad(self, req: bool):
        """Enable gradient computation on all parameters.

        Args:
            req (bool): Wether to require or not.
        """

        # Enable gradients for non-dictionary parameters
        self.gamma.requires_grad_(req)
        self.Q_flat_.requires_grad_(req)
        self.R_flat_.requires_grad_(req)

        # Enable gradients for dictionary parameters
        for tensor in self.alphas.values():
            tensor.requires_grad_(req)

        for tensor in self.betas.values():
            tensor.requires_grad_(req)


def tril_from_flat(flat: torch.Tensor, n: int) -> torch.Tensor:
    """Generate the lower triangular matrix associated with flat tensor.

    Args:
        flat (torch.Tensor): Flat tehsnro
        n (int): Dimension of the matrix.

    Raises:
        ValueError: Error if the the dimensions do not allow matrix computation.
        RuntimeError: Error if the computation fails.

    Returns:
        torch.Tensor: The lower triangular matrix.
    """

    if flat.numel() != (n * (n + 1)) // 2:
        raise ValueError("Incompatible dimensions for lower triangular matrix")

    L = torch.zeros(n, n, dtype=flat.dtype).index_put_(
        tuple(torch.tril_indices(n, n)), flat
    )

    return L


def flat_from_tril(L: torch.Tensor) -> torch.Tensor:
    """Flatten the lower triangular part (including the diagonal) of a square matrix L
    into a 1D tensor, in row-wise order.

    Args:
        L (torch.Tensor): Square lower-triangular matrix of shape (n, n).

    Raises:
        ValueError: If the input is not square.
        RuntimeError: If the flattening fails.

    Returns:
        torch.Tensor: Flattened 1D tensor containing the lower triangular entries.
    """

    try:
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("Input must be a square matrix")

        n = L.shape[0]
        i, j = torch.tril_indices(n, n)

        return L[i, j]

    except Exception as e:
        raise RuntimeError(f"Failed to flatten matrix: {e}") from e


def precision_from_log_cholesky(L: torch.Tensor) -> torch.Tensor:
    """Computes the inverse covariance matrix from log Cholesky factor.

    Args:
        L (torch.Tensor): log Cholesky factor

    Raises:
        RuntimeError: Error if the computation fails.

    Returns:
        torch.Tensor: The inverse covariance (precision) matrix.
    """

    L.diagonal().exp_()
    P = L @ L.T

    return P


def log_cholesky_from_precision(P: torch.Tensor) -> torch.Tensor:
    """Computes the log Cholesky-like factor used in precision_from_log_cholesky.
    (with log-diagonal convention) from the inverse covariance matrix.

    Args:
        P (torch.Tensor): Precision matrix (positive definite).

    Raises:
        RuntimeError: Error if the computation fails.

    Returns:
        torch.Tensor: Lower-triangular matrix L such that L.diag().exp() @ L.diag().exp().T = P
    """

    try:
        L: torch.Tensor = cast(torch.Tensor, torch.linalg.cholesky(P))  # type: ignore
        L.diagonal().log_()

        return L

    except Exception as e:
        raise RuntimeError(f"Failed to invert precision matrix: {e}") from e


def log_cholesky_from_flat(
    flat: torch.Tensor, n: int, method: str = "full"
) -> torch.Tensor:
    """Computes log cholesky from flat tensor according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, either for full, diagonal or isotropic covariance matrix. Defaults to "full".

    Raises:
        ValueError: If the array is not flat.
        ValueError: If the number of parameters is inconsistent with n.
        ValueError: If the number of parameters does not equal one.

    Returns:
        torch.Tensor: _description_
    """

    if flat.ndim != 1:
        raise ValueError(f"flat should be flat, got shape {flat.shape}")

    match method:
        case "full":
            return tril_from_flat(flat, n)
        case "diag":
            if n != flat.numel():
                raise ValueError(
                    f"Inocrrect number of elements for flat, got {flat.numel()} but expected {n}"
                )
            return torch.diag(flat)
        case "ball":
            if 1 != flat.numel():
                f"Inocrrect number of elements for flat, got {flat.numel()} but expected {1}"
            return flat * torch.eye(n)
        case _:
            raise ValueError(f"Got method {method} unknown")

    return P


def build_buckets(
    trajectories: list[Traj],
) -> dict[tuple[int, int], tuple[torch.Tensor, ...]]:
    """Builds buckets from trajectories for user convenience.

    Args:
        trajectories (list[Traj]): The list of individual trajectories.

    Raises:
        RuntimeError: If the construction of the buckets fails.

    Returns:
        dict[tuple[int, int], tuple[torch.Tensor, ...]]: A dictionnary of transition keys with a triplet of tensors (idxs, t0, t1).
    """

    try:
        # Process each individual trajectory
        buckets: DefaultDict[tuple[int, int], list[list[Any]]] = defaultdict(
            lambda: [[], [], []]
        )

        for i, trajectory in enumerate(trajectories):
            for (t0, s0), (t1, s1) in zip(trajectory[:-1], trajectory[1:]):
                key = (s0, s1)
                buckets[key][0].append(i)
                buckets[key][1].append(t0)
                buckets[key][2].append(t1)

        processed_buckets = {
            key: (
                torch.tensor(vals[0], dtype=torch.int64),
                torch.tensor(vals[1], dtype=torch.float32),
                torch.tensor(vals[2], dtype=torch.float32),
            )
            for key, vals in buckets.items()
            if vals[0]  # skip empty
        }

        return processed_buckets

    except Exception as e:
        raise RuntimeError(f"Failed to construct buckets: {e}") from e
