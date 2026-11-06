from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

from .ansatz import Ansatz, Component, GlobalQCMPS
from .logger import Logger

@dataclass
class OptimizationResult:
    parameters: np.ndarray
    initial_energy: float
    final_energy: float
    n_function_evaluations: int
    n_iterations: int
    converged: bool
    message: str
    energy_history: list[float]

class Optimizer(ABC):
    def __init__(self, logger: Logger | None = None, log_every_n_steps: int = 1):
        self.logger = logger
        
        if log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be a positive integer.")
        
        self.log_every_n_steps = log_every_n_steps
        self.step_counter = 0

    def log(self, message: str):
        if self.logger is not None:
            self.logger.log(f"[Optimizer] {message}")
        else:
            pass
    
    def callback_intermediate_result(self, intermediate_result, constant_shift: float = 0.0):
        if self.step_counter % self.log_every_n_steps == 0:
            total_intermediate = intermediate_result.fun + constant_shift
            self.log(f"Total Energy: {total_intermediate:.12f}") 
        self.step_counter += 1

    @abstractmethod
    def optimize(
        self, 
        ansatz: Ansatz, 
        components: Component | list[Component], 
        method: str = "L-BFGS-B",
        options: dict | None = None
    ) -> OptimizationResult:
        pass
    
class GlobalOptimizer(Optimizer):
    def optimize(
        self,
        ansatz: Ansatz, 
        components: Component | list[Component],
        method: str = "L-BFGS-B",
        options: dict | None = None,
    ) -> OptimizationResult:
        
        self.step_counter = 0
        
        if isinstance(components, Component):
            components = [components]
        
        function = ansatz.energy_objective(components)
        x0 = np.concatenate([component.current_parameters() for component in components])
        
        shift = ansatz.nuclear_repulsion or 0.0
        
        def callback(intermediate_result):
            self.callback_intermediate_result(intermediate_result, constant_shift=shift)
            
        initial_energy = ansatz.energy()
            
        result = minimize(
            fun=function,
            x0=x0,
            method=method,
            options=options,
            callback=callback
        )
        
        self.log(f"Global optimization completed. Final energy: {result.fun + shift:.12f}")
        
        x = result.x
        
        offset = 0
        for component in components:
            n_params = component.n_parameters
            component.update_parameters(x[offset:offset + n_params])
            offset += n_params 
            
        return OptimizationResult(
            parameters=x,
            initial_energy=initial_energy,
            final_energy=float(result.fun),
            n_function_evaluations=result.nfev + 1,
            n_iterations=result.nit,
            converged=result.success,
            message=result.message,
            energy_history=[initial_energy, result.fun]
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
        components: Component | list[Component],
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
        
        local_options = dict(options or {})
        local_options.setdefault("maxiter", self.local_maxiter)
        
        current_energy = ansatz.energy()
        initial_energy = current_energy
        
        sweep_energies = [current_energy]
        
        total_nfev = 1
        total_nit = 0
        converged = False
        
        forward = list(enumerate(groups))
        backward = list(reversed(forward[1:-1]))
        
        sweep_order = forward + backward
    
        shift = ansatz.nuclear_repulsion or 0.0
        def callback(intermediate_result):
            self.callback_intermediate_result(intermediate_result, constant_shift=shift)
        
        for sweep_index in range(self.max_sweeps):
            
            self.log(f"Sweep {sweep_index + 1}/{self.max_sweeps} started.")
            
            energy_before_sweep = current_energy
            
            for group_index, group in sweep_order:

                group_label = self._group_label(group_index, len(groups))
                self.log(f"Optimizing {group_label.lower()}/{len(groups)}...")
                
                baseline = qcmps.current_parameters()
                x0 = baseline[group].copy()
                
                function = ansatz.energy_objective(qcmps, parameter_groups = group)
                
                result = minimize(
                    fun=function,
                    x0=x0,
                    method=method,
                    options=local_options,
                    callback=callback
                )
                
                total_nfev += result.nfev
                total_nit += result.nit
                
                candidate = baseline.copy()
                candidate[group] = result.x
                
                if np.isfinite(result.fun) and result.fun < current_energy:
                    qcmps.update_parameters(candidate)
                    current_energy = float(result.fun)
                else:
                    qcmps.update_parameters(baseline)
                    
                self.log(f"[Sweep {sweep_index + 1} {group_label}] Total energy: {current_energy + shift:.12f}")
            
            sweep_energies.append(current_energy)
            
            relative_change = abs(energy_before_sweep - current_energy) / max(1.0, abs(energy_before_sweep))
            
            self.log(f"[Sweep {sweep_index + 1}] Total energy: {current_energy + shift:.12f}, Relative change: {relative_change:.3e}")
            
            if relative_change < self.tolerance:
                self.log(f"Convergence reached after {sweep_index + 1} sweeps. Relative change: {relative_change:.2e}")
                converged = True
                break
        
        if converged:
            message = "Sweep optimization converged to the relative energy tolerance."
        else:
            message = "Maximum number of sweeps was reached."
            
        return OptimizationResult(
            parameters=qcmps.current_parameters(),
            initial_energy=initial_energy,
            final_energy=current_energy,
            n_function_evaluations=total_nfev,
            n_iterations=total_nit,
            converged=converged,
            message=message,
            energy_history=sweep_energies
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
            
