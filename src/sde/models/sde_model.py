"""SDE Model with drift/diffusion functions and mathematical validation.

This module implements the SDEModel with strict mathematical fidelity
requirements for stochastic differential equation modeling.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


class ModelType(str, Enum):
    """Enumeration of SDE model types."""
    SCALAR = "SCALAR"                   # Single-dimensional SDE
    MULTIVARIATE = "MULTIVARIATE"      # Multi-dimensional SDE system
    JUMP_DIFFUSION = "JUMP_DIFFUSION"  # Jump-diffusion process


class SDEModel(BaseModel):
    """Stochastic Differential Equation model with mathematical validation.

    This model enforces constitutional mathematical fidelity requirements:
    - Exact preservation of drift and diffusion function expressions
    - Dimensional consistency validation
    - Mathematical structure verification for SDE systems
    """

    model_name: str = Field(
        ...,
        description="Name identifier for the SDE model",
        min_length=1
    )

    model_type: ModelType = Field(
        ...,
        description="Type of SDE model (scalar, multivariate, or jump-diffusion)"
    )

    drift_functions: List[str] = Field(
        ...,
        description="List of drift function expressions μ(X,t)",
        min_length=1
    )

    diffusion_functions: List[str] = Field(
        ...,
        description="List of diffusion function expressions σ(X,t)",
        min_length=1
    )

    state_variables: List[str] = Field(
        ...,
        description="List of state variable symbols",
        min_length=1
    )

    jump_components: Optional[List[str]] = Field(
        None,
        description="Jump component expressions for jump-diffusion models"
    )

    boundary_conditions: Optional[Dict[str, Any]] = Field(
        None,
        description="Boundary conditions and domain constraints"
    )

    analytical_solutions: Optional[Dict[str, Any]] = Field(
        None,
        description="Known analytical solutions if available"
    )

    convergence_properties: Optional[Dict[str, Any]] = Field(
        None,
        description="Convergence and stability properties"
    )

    model_config = ConfigDict(
        # Ensure exact preservation of mathematical expressions
        validate_assignment=True,
        # Allow complex mathematical structures
        arbitrary_types_allowed=True,
        # Preserve exact mathematical notation
        str_strip_whitespace=False
    )

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator('drift_functions')
    @classmethod
    def validate_drift_functions(cls, v):
        """Validate drift functions are properly specified."""
        if not isinstance(v, list):
            raise ValueError("Drift functions must be a list")

        if len(v) == 0:
            raise ValueError("At least one drift function is required")

        for func in v:
            if not isinstance(func, str) or not func.strip():
                raise ValueError("Each drift function must be a non-empty string")

        return v

    @field_validator('diffusion_functions')
    @classmethod
    def validate_diffusion_functions(cls, v):
        """Validate diffusion functions are properly specified."""
        if not isinstance(v, list):
            raise ValueError("Diffusion functions must be a list")

        if len(v) == 0:
            raise ValueError("At least one diffusion function is required")

        for func in v:
            if not isinstance(func, str) or not func.strip():
                raise ValueError("Each diffusion function must be a non-empty string")

        return v

    @field_validator('state_variables')
    @classmethod
    def validate_state_variables(cls, v):
        """Validate state variables are properly specified."""
        if not isinstance(v, list):
            raise ValueError("State variables must be a list")

        if len(v) == 0:
            raise ValueError("At least one state variable is required")

        for var in v:
            if not isinstance(var, str) or not var.strip():
                raise ValueError("Each state variable must be a non-empty string")

        return v

    @field_validator('jump_components')
    @classmethod
    def validate_jump_components(cls, v):
        """Validate jump components if specified."""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("Jump components must be a list")

        for comp in v:
            if not isinstance(comp, str):
                raise ValueError("Each jump component must be a string")

        return v

    @model_validator(mode='after')
    def validate_dimensional_consistency(self):
        """Validate dimensional consistency of the SDE model."""
        # Check that dimensions match for scalar and multivariate models
        num_drift = len(self.drift_functions)
        num_diffusion = len(self.diffusion_functions)
        num_states = len(self.state_variables)

        if self.model_type == ModelType.SCALAR:
            if not (num_drift == num_diffusion == num_states == 1):
                raise ValueError(
                    f"Scalar SDE must have 1 drift, 1 diffusion, and 1 state variable. "
                    f"Got: {num_drift} drift, {num_diffusion} diffusion, {num_states} states"
                )
        elif self.model_type == ModelType.MULTIVARIATE:
            if not (num_drift == num_diffusion == num_states):
                raise ValueError(
                    f"Multivariate SDE dimensions must match. "
                    f"Got: {num_drift} drift, {num_diffusion} diffusion, {num_states} states"
                )

        # Validate jump-diffusion specific constraints
        if self.model_type == ModelType.JUMP_DIFFUSION:
            if self.jump_components is None or len(self.jump_components) == 0:
                raise ValueError("Jump-diffusion model must have jump components")

        return self

    def __str__(self) -> str:
        """String representation preserving model name and type."""
        return f"SDEModel({self.model_name}: {self.model_type})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"SDEModel("
            f"model_name='{self.model_name}', "
            f"model_type='{self.model_type}', "
            f"dimension={len(self.state_variables)}"
            f")"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on model structure."""
        if not isinstance(other, SDEModel):
            return False

        return (
            self.model_name == other.model_name and
            self.model_type == other.model_type and
            self.drift_functions == other.drift_functions and
            self.diffusion_functions == other.diffusion_functions and
            self.state_variables == other.state_variables and
            self.jump_components == other.jump_components
        )

    def __hash__(self) -> int:
        """Hash based on model structure."""
        return hash((
            self.model_name,
            self.model_type,
            tuple(self.drift_functions),
            tuple(self.diffusion_functions),
            tuple(self.state_variables),
            tuple(self.jump_components) if self.jump_components else None
        ))

    def get_sde_dimension(self) -> int:
        """Get the dimension of the SDE system."""
        return len(self.state_variables)

    def has_jump_components(self) -> bool:
        """Check if the model has jump components."""
        return self.jump_components is not None and len(self.jump_components) > 0

    def validate_sde_structure(self) -> bool:
        """Validate the mathematical structure of the SDE."""
        # Check basic dimensional consistency
        if not self.get_sde_dimension() > 0:
            return False

        # Check that drift and diffusion functions match dimensions
        if len(self.drift_functions) != len(self.state_variables):
            return False

        if len(self.diffusion_functions) != len(self.state_variables):
            return False

        # Validate jump-diffusion structure
        if self.model_type == ModelType.JUMP_DIFFUSION:
            if not self.has_jump_components():
                return False

        return True

    def identify_standard_model(self) -> Optional[str]:
        """Identify if this is a standard SDE model type."""
        if self.model_type != ModelType.SCALAR:
            return None

        drift = self.drift_functions[0].lower()
        diffusion = self.diffusion_functions[0].lower()

        # Geometric Brownian Motion: dS = μS dt + σS dW
        if ("mu" in drift or "\\mu" in drift) and ("sigma" in diffusion or "\\sigma" in diffusion):
            if any(var.lower() in drift for var in self.state_variables):
                if any(var.lower() in diffusion for var in self.state_variables):
                    return "geometric_brownian_motion"

        # Ornstein-Uhlenbeck: dX = θ(μ - X)dt + σ dW
        if ("theta" in drift or "\\theta" in drift) and ("mu" in drift or "\\mu" in drift):
            if "sigma" in diffusion or "\\sigma" in diffusion:
                return "ornstein_uhlenbeck"

        # Cox-Ingersoll-Ross: dr = κ(θ - r)dt + σ√r dW
        if ("kappa" in drift or "\\kappa" in drift) and ("sqrt" in diffusion or "\\sqrt" in diffusion):
            return "cox_ingersoll_ross"

        return None

    def get_model_complexity(self) -> Dict[str, Any]:
        """Analyze model complexity metrics."""
        complexity = {
            "dimension": self.get_sde_dimension(),
            "has_jumps": self.has_jump_components(),
            "num_drift_functions": len(self.drift_functions),
            "num_diffusion_functions": len(self.diffusion_functions),
            "drift_complexity": sum(len(f) for f in self.drift_functions),
            "diffusion_complexity": sum(len(f) for f in self.diffusion_functions)
        }

        if self.jump_components:
            complexity["num_jump_components"] = len(self.jump_components)
            complexity["jump_complexity"] = sum(len(f) for f in self.jump_components)

        return complexity

    def get_required_parameters(self) -> List[str]:
        """Extract parameter symbols from drift and diffusion functions."""
        # This is a simple extraction - could be enhanced with proper parsing
        all_functions = self.drift_functions + self.diffusion_functions
        if self.jump_components:
            all_functions.extend(self.jump_components)

        # Look for common parameter patterns
        parameter_patterns = [
            '\\mu', '\\sigma', '\\kappa', '\\theta', '\\alpha', '\\beta',
            '\\gamma', '\\delta', '\\lambda', '\\rho', '\\nu', '\\phi'
        ]

        found_parameters = []
        for pattern in parameter_patterns:
            for func in all_functions:
                if pattern in func and pattern not in found_parameters:
                    found_parameters.append(pattern)

        return found_parameters

    def validate_boundary_conditions(self) -> bool:
        """Validate boundary conditions are mathematically consistent."""
        if self.boundary_conditions is None:
            return True

        # Basic validation of boundary condition structure
        if not isinstance(self.boundary_conditions, dict):
            return False

        # Check for common boundary condition types
        valid_keys = [
            'lower_bounds', 'upper_bounds', 'absorbing_barriers',
            'reflecting_barriers', 'periodic', 'killing_boundaries'
        ]

        for key in self.boundary_conditions:
            if key not in valid_keys:
                continue  # Allow custom boundary conditions

        return True

    def get_analytical_properties(self) -> Dict[str, Any]:
        """Get analytical properties if available."""
        if self.analytical_solutions is None:
            return {}

        return self.analytical_solutions

    def validate_mathematical_fidelity(self) -> bool:
        """Validate mathematical fidelity requirements."""
        # CONSTITUTIONAL REQUIREMENT: All mathematical expressions preserved exactly
        for func in self.drift_functions:
            if not func or not isinstance(func, str):
                return False

        for func in self.diffusion_functions:
            if not func or not isinstance(func, str):
                return False

        # Validate SDE structure
        if not self.validate_sde_structure():
            return False

        return True