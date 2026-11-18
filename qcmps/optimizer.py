from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Callable

from .ansatz import Ansatz, Component, GlobalQCMPS
from .logger import Logger

@dataclass
class OptimizationResult:
    parameters: np.ndarray
    initial_value: float
    final_value: float
    n_function_evaluations: int
    n_iterations: int
    converged: bool
    message: str
    fun_values_history: list[float]

class Optimizer(ABC):
    def __init__(self, logger: Logger | None = None, log_every_n_steps: int | None = 1):
        self.logger = logger
        
        if log_every_n_steps is not None and log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be a positive integer.")
        
        self.log_every_n_steps = log_every_n_steps
        self.step_counter = 0

    def log_message(self, message: str):
        if self.logger is not None:
            self.logger.log(f"[Optimizer] {message}")
        
    def log_value(self, metric: str, value: float, delta_funev: int = 1):
        if self.logger is not None:
            self.logger.log_value(metric, value, delta_funev)
        
    def callback_intermediate_result(self, intermediate_result, label: str = "Objective value"):
        if self.log_every_n_steps is None:
            return
        
        if self.step_counter % self.log_every_n_steps == 0:
            self.log_message(f"{label}: {intermediate_result.fun:.12f}") 
            
        self.step_counter += 1

    @abstractmethod
    def optimize(
        self, 
        ansatz: Ansatz, 
        function: Callable[[Ansatz], float],
        components: Component | list[Component], 
        method: str = "L-BFGS-B",
        options: dict | None = None
    ) -> OptimizationResult:
        pass
    
class GlobalOptimizer(Optimizer):
    def optimize(
        self,
        ansatz: Ansatz, 
        function: Callable[[Ansatz], float],
        components: Component | list[Component],
        method: str = "L-BFGS-B",
        options: dict | None = None,
    ) -> OptimizationResult:
        
        self.step_counter = 0
        
        if isinstance(components, Component):
            components = [components]
        
        objective = ansatz.objective(function, components)
        x0 = np.concatenate([component.current_parameters() for component in components])
        
        function_label = getattr(function, "label", "Objective value")
        
        def callback(intermediate_result):
            self.callback_intermediate_result(intermediate_result, label=function_label)
            
        initial_fun_value = function(ansatz)
            
        result = minimize(
            fun=objective,
            x0=x0,
            method=method,
            options=options,
            callback=callback
        )
        
        x = result.x
        
        offset = 0
        for component in components:
            n_params = component.n_parameters
            component.update_parameters(x[offset:offset + n_params])
            offset += n_params 
            
        return OptimizationResult(
            parameters=x,
            initial_value=initial_fun_value,
            final_value=float(result.fun),
            n_function_evaluations=result.nfev + 1,
            n_iterations=result.nit,
            converged=result.success,
            message=result.message,
            fun_values_history=[initial_fun_value, result.fun]
        )
             
class SweepOptimizer(Optimizer):
    
    def __init__(self, logger: Logger | None = None, log_every_n_steps: int = 1, max_sweeps = 10, local_maxiter = 10, tolerance = 1e-7):
        
        super().__init__(logger, log_every_n_steps)
        self.max_sweeps = max_sweeps
        self.local_maxiter = local_maxiter
        self.tolerance = tolerance

    @abstractmethod
    def _parameter_groups(self, qcmps: GlobalQCMPS) -> list[np.ndarray]:
        pass
    
    @abstractmethod
    def _group_label(self, group_index: int, n_groups: int) -> str:
        pass

    def optimize(
        self, 
        ansatz: Ansatz, 
        function: Callable[[Ansatz], float],
        components: Component | list[Component],
        active_parameters: np.ndarray | list[np.ndarray] | None = None,
        method: str = "L-BFGS-B",
        options: dict | None = None,
    ) -> OptimizationResult:
        
        self.step_counter = 0
        
        if not isinstance(components, GlobalQCMPS):
            raise TypeError(f"{type(self).__name__} only supports GlobalQCMPS components.")
        
        qcmps: GlobalQCMPS = components
        
        if not any(component is qcmps for component in ansatz.components):
            raise ValueError("The specified component is not part of the ansatz.")
        
        groups = self._parameter_groups(qcmps)
        
        if not groups:
            raise ValueError("No parameter groups found in the GlobalQCMPS component.")
        
        if active_parameters is not None:
            
            if isinstance(active_parameters, list):
                active_parameters = np.concatenate(active_parameters)
            
            active_parameters = np.asarray(active_parameters, dtype = int)
            
            if active_parameters.ndim != 1:
                raise ValueError("active_parameters must be a one-dimensional array.")
            
            if np.any(active_parameters < 0) or np.any(active_parameters >= qcmps.n_parameters):
                raise ValueError("Active parameter index is out of range")
            
            if len(np.unique(active_parameters)) != len(active_parameters):
                raise ValueError("Active parameter indices must be unique.")
            
            active_mask = np.zeros(qcmps.n_parameters, dtype=bool)
            active_mask[active_parameters] = True
            
            groups = [group[active_mask[group]] for group in groups]
            
            groups = [group for group in groups if len(group) > 0]
        
        local_options = dict(options or {})
        local_options.setdefault("maxiter", self.local_maxiter)
        
        current_fun_value = function(ansatz)
        initial_fun_value = current_fun_value
        
        sweep_fun_values = [current_fun_value]
        
        total_nfev = 1
        total_nit = 0
        converged = False
        
        forward = list(enumerate(groups))
        backward = list(reversed(forward[1:-1]))
        sweep_order = forward + backward
        
        function_label = getattr(function, "label", "Objective value")
        
        for sweep_index in range(self.max_sweeps):
            
            self.log_message(f"Sweep {sweep_index + 1}/{self.max_sweeps} started.")
            fun_value_before_sweep = current_fun_value
            
            for group_index, group in sweep_order:

                group_label = self._group_label(group_index, len(groups))
                
                baseline = qcmps.current_parameters()
                x0 = baseline[group].copy()
                
                objective = ansatz.objective(function, qcmps, parameter_groups = group)
                
                result = minimize(
                    fun=objective,
                    x0=x0,
                    method=method,
                    options=local_options,
                )
                
                total_nfev += result.nfev
                total_nit += result.nit
                
                candidate = baseline.copy()
                candidate[group] = result.x
                
                if np.isfinite(result.fun) and result.fun < current_fun_value:
                    qcmps.update_parameters(candidate)
                    current_fun_value = float(result.fun)
                else:
                    qcmps.update_parameters(baseline)
                    
                self.log_message(f"[Sweep {sweep_index + 1} {group_label}] {function_label}: {current_fun_value:.12f}")
                self.log_value(function_label, current_fun_value, result.nfev + 1)
                
            
            sweep_fun_values.append(current_fun_value)
            
            relative_change = abs(fun_value_before_sweep - current_fun_value) / max(1.0, abs(fun_value_before_sweep))
            
            self.log_message(f"[Sweep {sweep_index + 1}] {function_label}: {current_fun_value:.12f}, Relative change: {relative_change:.3e}")
            
            if relative_change < self.tolerance:
                self.log_message(f"Convergence reached after {sweep_index + 1} sweeps. Relative change: {relative_change:.2e}")
                converged = True
                break
        
        if converged:
            message = f"Sweep optimization converged to the relative {function_label} tolerance."
        else:
            message = "Maximum number of sweeps was reached."
            
        return OptimizationResult(
            parameters=qcmps.current_parameters(),
            initial_value=initial_fun_value,
            final_value=current_fun_value,
            n_function_evaluations=total_nfev,
            n_iterations=total_nit,
            converged=converged,
            message=message,
            fun_values_history=sweep_fun_values
        )

class OneSiteOptimizer(SweepOptimizer):
    
    def _parameter_groups(self, qcmps: GlobalQCMPS) -> list[np.ndarray]:
        return qcmps.block_parameter_groups()

    def _group_label(self, group_index: int, n_groups: int) -> str:
        return f"Block {group_index + 1}"
    
class TwoSiteOptimizer(SweepOptimizer):

    def _parameter_groups(self, qcmps: GlobalQCMPS) -> list[np.ndarray]:
        block_groups = qcmps.block_parameter_groups()

        if len(block_groups) < 2:
            raise ValueError("TwoSiteOptimizer requires at least two QCMPS parameter blocks.")

        return [np.concatenate((left_group, right_group)) for left_group, right_group in zip(block_groups[:-1], block_groups[1:])]

    def _group_label(self, group_index: int, n_groups: int) -> str:
        return f"Site pair {group_index + 1}-{group_index + 2}"
            
class SPSAOptimizer(Optimizer):
    def __init__(
        self,
        lerning_rate: float,
        perturbation: float,
        maxiter: int = 1000,
        alpha: float = 0.602,
        gamma: float = 0.101,
        tolerance: float | None = None,
        patience: int = 10,
        seed: int | None = None,
        logger: Logger | None = None,
        log_every_n_steps: int | None = 100,
    ):
        super().__init__(logger, log_every_n_steps)
        self.learning_rate = lerning_rate
        self.perturbation = perturbation
        self.maxiter = maxiter
        self.alpha = alpha
        self.gamma = gamma
        self.tolerance = tolerance
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        
    def optimize(
        self,
        ansatz: Ansatz,
        function: Callable[[Ansatz], float],
        components: Component | list[Component],
    ) -> OptimizationResult:
        
        self.step_counter = 0
        
        if isinstance(components, Component):
            components = [components]
            
        if not components:
            raise ValueError("At least one component must be specified for optimization.")
        
        objective = ansatz.objective(function, components)
        
        x = np.concatenate([component.current_parameters() for component in components])
        
        function_label = getattr(function, "label", "Objective value")
        
        initial_fun_value = objective(x)
        current_fun_value = initial_fun_value
        
        history = [initial_fun_value]
        n_function_evaluations = 1
        small_step_count = 0
        converged = False
        completed_iterations = 0
        
        for iteration in range(self.maxiter):
            k = iteration + 1
            
            learning_rate = self.learning_rate / (k ** self.alpha)
            perturbation = self.perturbation / (k ** self.gamma)
            
            delta = self.rng.choice((-1.0, 1.0), size=len(x))
            
            value_plus = objective(x + perturbation * delta)
            value_minus = objective(x - perturbation * delta)
            
            n_function_evaluations += 2
            
            gradient = delta * (value_plus - value_minus) / (2 * perturbation)
            
            parameter_step = -learning_rate * gradient
            x += parameter_step
            
            completed_iterations = k
            
            if self.tolerance is not None:
                normalized_step = np.linalg.norm(parameter_step) / np.sqrt(len(x))
                
                if normalized_step < self.tolerance:
                    small_step_count += 1
                else:
                    small_step_count = 0
                    
                if small_step_count >= self.patience:
                    converged = True
                    break
                
            should_log = self.log_every_n_steps is not None and k % self.log_every_n_steps == 0
            
            if should_log:
                current_fun_value = objective(x)
                history.append(current_fun_value)
                
                self.log_message(f"{function_label}: {current_fun_value:.12f}")
        
        final_fun_value = objective(x)
        n_function_evaluations += 1
        
        if not history or history[-1] != final_fun_value:
            history.append(final_fun_value)
            
        if converged:
            message = f"SPSA optimization converged after {completed_iterations} iterations."
        else:
            message = f"SPSA optimization reached the maximum number of iterations ({self.maxiter})."
            
        return OptimizationResult(
            parameters=x.copy(),
            initial_value=initial_fun_value,
            final_value=final_fun_value,
            n_function_evaluations=n_function_evaluations,
            n_iterations=completed_iterations,
            converged=converged,
            message=message,
            fun_values_history=history
        )    