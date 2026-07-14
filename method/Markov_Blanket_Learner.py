import numpy as np
import pandas as pd
from typing import Dict, List, Union

from Utils.CI_test import CI_test


Variable = Union[str, int]


class MarkovBlanketLearner:
    """Shared validation and cache support for CI-based Markov blanket learning."""

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        ci_test: CI_test = None,
        alpha: float = 0.05,
        **kwargs,
    ):
        self._validate_data(data)
        self.data = data

        if ci_test is None:
            ci_test_type = kwargs.get("ci_test_type")
            if ci_test_type is None:
                raise ValueError("ci_test_type must be provided when ci_test is None.")
            self.ci_test = CI_test(data, method=ci_test_type, alpha=alpha, **kwargs)
        elif isinstance(ci_test, CI_test):
            self.ci_test = ci_test
        else:
            raise TypeError("ci_test must be an instance of CI_test.")

        self.num_nodes = data.shape[1]
        self.mb_set: Dict[Variable, List[Variable]] = {}
        self.latent_variables = kwargs.get("latent_nodes")
        self.selection_bias_nodes = kwargs.get("selection_bias_nodes")

        hidden_variables = set(self.latent_variables or []) | set(
            self.selection_bias_nodes or []
        )
        if hidden_variables:
            if str(self.ci_test.method).lower() != "d_sep":
                raise ValueError(
                    "Only the d_sep method supports latent variables or selection bias."
                )
            if self.latent_variables is not None and not isinstance(
                self.latent_variables, list
            ):
                raise TypeError("latent_nodes must be a list of column labels or indices.")
            if self.selection_bias_nodes is not None and not isinstance(
                self.selection_bias_nodes, list
            ):
                raise TypeError(
                    "selection_bias_nodes must be a list of column labels or indices."
                )
            unknown = hidden_variables - set(self.column_names)
            if unknown:
                raise ValueError(f"Hidden variables not found in data columns: {unknown}")
            if set(self.latent_variables or []) & set(self.selection_bias_nodes or []):
                raise ValueError(
                    "latent_nodes and selection_bias_nodes must not overlap."
                )

        self.observed_variables = [
            variable for variable in self.column_names if variable not in hidden_variables
        ]
        self.bool_mb_df = pd.DataFrame(
            0, index=self.observed_variables, columns=self.observed_variables
        )

    def _validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Validate the input matrix and record its variable names."""
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError("Data must be a pandas DataFrame or a NumPy array.")
        if data.ndim != 2:
            raise ValueError("data must be a two-dimensional matrix.")
        if isinstance(data, pd.DataFrame):
            self.column_names = list(data.columns)
        else:
            self.column_names = list(range(data.shape[1]))

    def _format_target(self, target: Variable) -> Variable:
        """Validate and normalize a target variable identifier."""
        if isinstance(target, str):
            if target not in self.column_names:
                raise ValueError(f"Target '{target}' not found in data columns.")
            return target
        if isinstance(target, int):
            if target not in self.column_names:
                raise ValueError(f"Target index {target} is out of bounds.")
            return target
        raise TypeError("Target must be a string label or an integer index.")

    def _update_membership(self, target: Variable, mb: List[Variable]) -> None:
        """Cache symmetric Markov blanket membership decisions."""
        for variable in self.observed_variables:
            if variable == target:
                continue
            value = 1 if variable in mb else -1
            self.bool_mb_df.loc[target, variable] = value
            self.bool_mb_df.loc[variable, target] = value

    def _get_known_members(self, target: Variable) -> List[Variable]:
        """Return variables already known to belong to the target blanket."""
        return [
            variable
            for variable in self.bool_mb_df.columns
            if self.bool_mb_df.loc[target, variable] == 1
        ]


class TC_learn(MarkovBlanketLearner):
    """Learn a Markov blanket by total conditioning."""

    def __call__(self, target: Variable, **kwargs) -> List[Variable]:
        return self.get_markov_blanket(target)

    def get_markov_blanket(self, target: Variable) -> List[Variable]:
        """Return the target blanket, reusing previously cached decisions."""
        target = self._format_target(target)
        if target in self.mb_set:
            return self.mb_set[target]
        if target not in self.observed_variables:
            raise ValueError(f"Target {target} is not an observed variable.")

        candidates = [
            variable for variable in self.observed_variables if variable != target
        ]
        mb = self._get_known_members(target)
        for variable in candidates:
            if variable in mb or self.bool_mb_df.loc[target, variable] == -1:
                continue
            conditioning_set = [
                candidate for candidate in candidates if candidate != variable
            ]
            independent, _ = self.ci_test(target, variable, conditioning_set)
            if not independent:
                mb.append(variable)

        self.mb_set[target] = mb
        self._update_membership(target, mb)
        return mb


class gaussian_MB_learn:
    """Learn Gaussian Markov blankets from thresholded partial correlations."""

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        alpha: float = None,
        **kwargs,
    ):
        from scipy import stats

        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError("Data must be a pandas DataFrame or a NumPy array.")

        self.num_samples, self.num_nodes = data.shape
        if self.num_samples <= self.num_nodes + 1:
            raise ValueError(
                "Number of samples must exceed the number of variables plus one."
            )

        if isinstance(data, pd.DataFrame):
            values = data.to_numpy()
            self.column_names = list(data.columns)
        else:
            values = data
            self.column_names = list(range(data.shape[1]))

        correlation = np.corrcoef(values, rowvar=False)
        precision = np.linalg.pinv(correlation)
        normalization = np.sqrt(np.diag(precision))
        partial_correlation = np.abs(
            precision / normalization[:, None] / normalization[None, :]
        )

        significance = 1 / self.num_nodes**2 if alpha is None else alpha
        threshold = np.tanh(
            stats.norm.ppf(1 - significance / 2)
            / np.sqrt(self.num_samples - self.num_nodes - 1)
        )
        membership = np.where(partial_correlation > threshold, 1, -1)
        np.fill_diagonal(membership, 0)
        self.bool_mb_df = pd.DataFrame(
            membership, index=self.column_names, columns=self.column_names
        )

    def __call__(self, target: Variable, **kwargs) -> List[Variable]:
        """Return variables selected for the target Markov blanket."""
        if target not in self.bool_mb_df.index:
            raise ValueError(f"Target '{target}' not found in data columns.")
        return [
            variable
            for variable in self.bool_mb_df.columns
            if self.bool_mb_df.loc[target, variable] == 1
        ]


def MB_learn(
    data: Union[pd.DataFrame, np.ndarray],
    ci_test: CI_test = None,
    alpha: float = None,
    **kwargs,
):
    """Create the Markov blanket learner used by a LoCaLS execution path."""
    method = kwargs.get("mb_method_type", "TC")
    if method == "TC":
        return TC_learn(data, ci_test, alpha, **kwargs)
    if method == "gaussian_MB":
        return gaussian_MB_learn(data, alpha, **kwargs)
    raise ValueError(
        f"Unknown Markov blanket method '{method}'. Supported methods: TC, gaussian_MB."
    )
