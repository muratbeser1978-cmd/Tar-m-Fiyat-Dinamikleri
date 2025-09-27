"""Model Parameter with economic constraints and calibration metadata.

This module implements the ModelParameter model with strict mathematical fidelity
and comprehensive economic modeling capabilities.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


class ParameterType(str, Enum):
    """Enumeration of parameter types for economic models."""
    STRUCTURAL = "STRUCTURAL"       # Structural economic parameters
    BEHAVIORAL = "BEHAVIORAL"       # Behavioral parameters
    TECHNOLOGICAL = "TECHNOLOGICAL" # Technology parameters
    POLICY = "POLICY"              # Policy parameters
    STOCHASTIC = "STOCHASTIC"      # Noise/volatility parameters
    CALIBRATED = "CALIBRATED"      # Externally calibrated parameters


class ModelParameter(BaseModel):
    """Model parameter with economic constraints and calibration data.

    This model enforces constitutional mathematical fidelity requirements:
    - Exact preservation of parameter symbols and mathematical expressions
    - Economic constraint validation and calibration metadata
    - Parameter relationship tracking and sensitivity analysis
    """

    name: str = Field(
        ...,
        description="Parameter name for identification",
        min_length=1
    )

    symbol: str = Field(
        ...,
        description="Mathematical symbol (LaTeX notation preserved exactly)",
        min_length=1
    )

    description: str = Field(
        ...,
        description="Economic interpretation and meaning",
        min_length=1
    )

    units: str = Field(
        ...,
        description="Physical or economic units",
        min_length=1
    )

    default_value: float = Field(
        ...,
        description="Default parameter value"
    )

    parameter_type: ParameterType = Field(
        ...,
        description="Type of parameter for economic classification"
    )

    value_range: Optional[List[float]] = Field(
        None,
        description="Valid range [min, max] for parameter values",
        min_length=2,
        max_length=2
    )

    economic_interpretation: Optional[Dict[str, Any]] = Field(
        None,
        description="Economic meaning and context"
    )

    calibration_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Parameter calibration and estimation metadata"
    )

    prior_distribution: Optional[Dict[str, Any]] = Field(
        None,
        description="Prior distribution for Bayesian estimation"
    )

    related_parameters: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Related parameters and their relationships"
    )

    constraints: Optional[List[str]] = Field(
        None,
        description="Mathematical constraints on parameter values"
    )

    sensitivity_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Sensitivity analysis results and metrics"
    )

    model_config = ConfigDict(
        # Ensure exact preservation of all fields
        validate_assignment=True,
        # Allow for complex economic data structures
        arbitrary_types_allowed=True,
        # Preserve exact mathematical notation
        str_strip_whitespace=False
    )

    @field_validator('name', 'symbol', 'description', 'units')
    @classmethod
    def validate_required_strings(cls, v):
        """Validate that required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Required string field cannot be empty")
        return v.strip()

    @field_validator('value_range')
    @classmethod
    def validate_value_range(cls, v):
        """Validate parameter value range constraints."""
        if v is None:
            return v

        if len(v) != 2:
            raise ValueError("Value range must have exactly 2 elements [min, max]")

        min_val, max_val = v[0], v[1]
        if min_val > max_val:
            raise ValueError("Value range minimum must be <= maximum")

        return v

    @model_validator(mode='after')
    def validate_default_value_in_range(self):
        """Validate that default value is within specified range."""
        if self.value_range is not None:
            min_val, max_val = self.value_range[0], self.value_range[1]
            if not (min_val <= self.default_value <= max_val):
                raise ValueError(
                    f"Default value {self.default_value} is outside range [{min_val}, {max_val}]"
                )
        return self

    @field_validator('economic_interpretation')
    @classmethod
    def validate_economic_interpretation(cls, v):
        """Validate economic interpretation structure."""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("Economic interpretation must be a dictionary")

        return v

    @field_validator('calibration_info')
    @classmethod
    def validate_calibration_info(cls, v):
        """Validate calibration information structure."""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("Calibration info must be a dictionary")

        return v

    @field_validator('prior_distribution')
    @classmethod
    def validate_prior_distribution(cls, v):
        """Validate prior distribution specification."""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("Prior distribution must be a dictionary")

        # Validate required fields for distribution
        if 'distribution_type' in v:
            valid_distributions = ['normal', 'lognormal', 'beta', 'gamma', 'uniform']
            if v['distribution_type'] not in valid_distributions:
                raise ValueError(f"Invalid distribution type: {v['distribution_type']}")

        return v

    @field_validator('related_parameters')
    @classmethod
    def validate_related_parameters(cls, v):
        """Validate related parameters structure."""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("Related parameters must be a list")

        for param_rel in v:
            if not isinstance(param_rel, dict):
                raise ValueError("Each related parameter must be a dictionary")

        return v

    @field_validator('constraints')
    @classmethod
    def validate_constraints(cls, v):
        """Validate mathematical constraints."""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("Constraints must be a list")

        for constraint in v:
            if not isinstance(constraint, str):
                raise ValueError("Each constraint must be a string")

        return v

    @field_validator('sensitivity_analysis')
    @classmethod
    def validate_sensitivity_analysis(cls, v):
        """Validate sensitivity analysis structure."""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("Sensitivity analysis must be a dictionary")

        return v

    def __str__(self) -> str:
        """String representation preserving mathematical symbol."""
        return f"ModelParameter({self.symbol}: {self.description} = {self.default_value})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"ModelParameter("
            f"name='{self.name}', "
            f"symbol='{self.symbol}', "
            f"default_value={self.default_value}, "
            f"parameter_type='{self.parameter_type}'"
            f")"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on name and mathematical properties."""
        if not isinstance(other, ModelParameter):
            return False

        return (
            self.name == other.name and
            self.symbol == other.symbol and
            self.description == other.description and
            self.units == other.units and
            self.default_value == other.default_value and
            self.parameter_type == other.parameter_type and
            self.value_range == other.value_range
        )

    def __hash__(self) -> int:
        """Hash based on parameter name and symbol."""
        return hash((
            self.name,
            self.symbol,
            self.default_value,
            self.parameter_type,
            tuple(self.value_range) if self.value_range else None
        ))

    def __lt__(self, other) -> bool:
        """Less than comparison based on default values for ordering."""
        if not isinstance(other, ModelParameter):
            return NotImplemented
        return self.default_value < other.default_value

    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, ModelParameter):
            return NotImplemented
        return self.default_value <= other.default_value

    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if not isinstance(other, ModelParameter):
            return NotImplemented
        return self.default_value > other.default_value

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        if not isinstance(other, ModelParameter):
            return NotImplemented
        return self.default_value >= other.default_value

    def is_in_range(self, value: float) -> bool:
        """Check if a value is within the parameter's valid range."""
        if self.value_range is None:
            return True
        return self.value_range[0] <= value <= self.value_range[1]

    def get_economic_meaning(self) -> str:
        """Get economic interpretation of the parameter."""
        if self.economic_interpretation and 'economic_meaning' in self.economic_interpretation:
            return self.economic_interpretation['economic_meaning']
        return self.description

    def get_calibration_method(self) -> Optional[str]:
        """Get the calibration method used for this parameter."""
        if self.calibration_info and 'method' in self.calibration_info:
            return self.calibration_info['method']
        return None

    def get_standard_error(self) -> Optional[float]:
        """Get the standard error from calibration."""
        if self.calibration_info and 'standard_error' in self.calibration_info:
            return self.calibration_info['standard_error']
        return None

    def get_confidence_interval(self) -> Optional[List[float]]:
        """Get confidence interval from calibration."""
        if self.calibration_info and 'confidence_interval' in self.calibration_info:
            return self.calibration_info['confidence_interval']
        return None

    def validate_economic_constraints(self) -> bool:
        """Validate parameter against economic constraints."""
        # Check value is in range
        if not self.is_in_range(self.default_value):
            return False

        # Check mathematical constraints
        if self.constraints:
            # Basic constraint validation (could be extended with symbolic math)
            for constraint in self.constraints:
                if self.symbol in constraint:
                    # Simple positivity check
                    if ">" in constraint and "0" in constraint:
                        if self.default_value <= 0:
                            return False

        return True

    def update_value(self, new_value: float) -> 'ModelParameter':
        """Create a copy of the parameter with updated value."""
        if not self.is_in_range(new_value):
            raise ValueError(f"Value {new_value} is outside valid range {self.value_range}")

        return self.model_copy(update={"default_value": new_value})

    def add_related_parameter(self, parameter_symbol: str, relationship: str, constraint: str = None) -> None:
        """Add a related parameter relationship."""
        if self.related_parameters is None:
            self.related_parameters = []

        relationship_info = {
            "parameter": parameter_symbol,
            "relationship": relationship
        }

        if constraint:
            relationship_info["constraint"] = constraint

        self.related_parameters.append(relationship_info)

    def get_sensitivity_metrics(self) -> Dict[str, float]:
        """Get sensitivity analysis metrics."""
        if not self.sensitivity_analysis:
            return {}

        metrics = {}
        if 'local_sensitivity' in self.sensitivity_analysis:
            metrics.update(self.sensitivity_analysis['local_sensitivity'])
        if 'global_sensitivity' in self.sensitivity_analysis:
            metrics.update(self.sensitivity_analysis['global_sensitivity'])

        return metrics