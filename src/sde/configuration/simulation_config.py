"""Simulation Configuration model with parameter validation.

This module implements the SimulationConfiguration model with comprehensive
validation for SDE simulation parameters and constraints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class NumericalScheme(str, Enum):
    """Enumeration of numerical schemes for SDE simulation."""
    EULER_MARUYAMA = "EULER_MARUYAMA"
    MILSTEIN = "MILSTEIN"
    STOCHASTIC_RUNGE_KUTTA = "STOCHASTIC_RUNGE_KUTTA"


class SimulationConfiguration(BaseModel):
    """Simulation configuration with comprehensive parameter validation.

    This model enforces constitutional mathematical fidelity requirements:
    - Exact preservation of simulation parameters
    - Mathematical constraint validation
    - Numerical stability requirements
    """

    model_reference: str = Field(
        ...,
        description="Reference to the SDE model to simulate",
        min_length=1
    )

    time_horizon: float = Field(
        ...,
        description="Total simulation time",
        gt=0
    )

    time_steps: int = Field(
        ...,
        description="Number of time steps",
        gt=0
    )

    monte_carlo_paths: int = Field(
        ...,
        description="Number of Monte Carlo simulation paths",
        gt=0
    )

    numerical_scheme: NumericalScheme = Field(
        ...,
        description="Numerical scheme for SDE integration"
    )

    random_seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )

    performance_targets: Optional[Dict[str, Any]] = Field(
        None,
        description="Performance targets and constraints"
    )

    output_specifications: Optional[List[str]] = Field(
        None,
        description="Output data specifications"
    )

    parallelization_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Parallelization configuration"
    )

    model_config = ConfigDict(
        # Ensure exact preservation of simulation parameters
        validate_assignment=True,
        # Allow complex configuration structures
        arbitrary_types_allowed=True
    )

    @field_validator('model_reference')
    @classmethod
    def validate_model_reference(cls, v):
        """Validate model reference is not empty."""
        if not v or not v.strip():
            raise ValueError("Model reference cannot be empty")
        return v.strip()

    @field_validator('time_horizon')
    @classmethod
    def validate_time_horizon(cls, v):
        """Validate time horizon is positive."""
        if v <= 0:
            raise ValueError("Time horizon must be positive")
        return v

    @field_validator('time_steps')
    @classmethod
    def validate_time_steps(cls, v):
        """Validate time steps is positive."""
        if v <= 0:
            raise ValueError("Number of time steps must be positive")
        return v

    @field_validator('monte_carlo_paths')
    @classmethod
    def validate_monte_carlo_paths(cls, v):
        """Validate Monte Carlo paths is positive."""
        if v <= 0:
            raise ValueError("Number of Monte Carlo paths must be positive")
        return v

    def __str__(self) -> str:
        """String representation of simulation configuration."""
        return f"SimulationConfig({self.model_reference}, T={self.time_horizon}, paths={self.monte_carlo_paths})"

    def get_time_step_size(self) -> float:
        """Calculate time step size dt = T/N."""
        return self.time_horizon / self.time_steps

    def validate_stability_conditions(self) -> bool:
        """Validate numerical stability conditions."""
        dt = self.get_time_step_size()

        # Basic stability check for explicit schemes
        if dt > 0.1:  # Very conservative check
            return False

        return True

    def get_total_sample_points(self) -> int:
        """Get total number of sample points generated."""
        return self.monte_carlo_paths * (self.time_steps + 1)

    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimate: 8 bytes per float * total sample points
        return (self.get_total_sample_points() * 8) / (1024 * 1024)

    def validate_constitutional_compliance(self) -> bool:
        """Validate constitutional simulation requirements."""
        # CONSTITUTIONAL REQUIREMENT: All parameters must be positive and valid
        if self.time_horizon <= 0 or self.time_steps <= 0 or self.monte_carlo_paths <= 0:
            return False

        return True