"""Mathematical Variable model with Pydantic validation.

This module implements the MathematicalVariable model with strict mathematical fidelity
requirements, ensuring exact preservation of mathematical symbols and notation.
"""

from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum


class VariableType(str, Enum):
    """Enumeration of mathematical variable types."""
    STATE = "STATE"
    ALGEBRAIC = "ALGEBRAIC"
    PARAMETER = "PARAMETER"
    CONSTANT = "CONSTANT"


class MathematicalVariable(BaseModel):
    """Mathematical variable with complete metadata and validation rules.

    This model enforces constitutional mathematical fidelity requirements:
    - Exact preservation of mathematical symbols without modification
    - Complete metadata capture for mathematical context
    - Validation of mathematical constraints and relationships
    """

    symbol: str = Field(
        ...,
        description="Mathematical symbol (LaTeX notation preserved exactly)",
        min_length=1
    )

    description: str = Field(
        ...,
        description="Human-readable description of the variable",
        min_length=1
    )

    units: str = Field(
        ...,
        description="Physical or mathematical units",
        min_length=1
    )

    variable_type: VariableType = Field(
        ...,
        description="Type of mathematical variable"
    )

    domain: Optional[List[Union[float, str]]] = Field(
        None,
        description="Domain constraints [min, max] where str can be 'inf' or '-inf'",
        min_length=2,
        max_length=2
    )

    initial_value: Optional[float] = Field(
        None,
        description="Initial value if applicable"
    )

    dependencies: List[str] = Field(
        default_factory=list,
        description="List of variables this variable depends on"
    )

    equation_refs: List[str] = Field(
        default_factory=list,
        description="References to equations containing this variable"
    )

    model_config = ConfigDict(
        # Ensure exact preservation of all fields
        validate_assignment=True,
        # Allow for mathematical infinity values
        arbitrary_types_allowed=True,
        # Preserve validation for mathematical consistency
        str_strip_whitespace=False  # Preserve exact mathematical notation
    )

    @field_validator('symbol')
    @classmethod
    def validate_symbol_preservation(cls, v):
        """Validate that mathematical symbols are preserved exactly."""
        if not v or not v.strip():
            raise ValueError("Mathematical symbol cannot be empty")

        # Preserve mathematical notation exactly - no modifications allowed
        return v.strip()

    @field_validator('domain')
    @classmethod
    def validate_domain_constraints(cls, v):
        """Validate mathematical domain constraints."""
        if v is None:
            return v

        if len(v) != 2:
            raise ValueError("Domain must have exactly 2 elements [min, max]")

        # Convert string representations of infinity
        converted_domain = []
        for val in v:
            if isinstance(val, str):
                if val.lower() in ['inf', 'infinity', '+inf']:
                    converted_domain.append(float('inf'))
                elif val.lower() in ['-inf', '-infinity']:
                    converted_domain.append(float('-inf'))
                else:
                    try:
                        converted_domain.append(float(val))
                    except ValueError:
                        raise ValueError(f"Invalid domain value: {val}")
            else:
                converted_domain.append(float(val))

        # Validate domain ordering
        if converted_domain[0] > converted_domain[1]:
            raise ValueError("Domain minimum must be <= maximum")

        return converted_domain

    @model_validator(mode='after')
    def validate_initial_value_in_domain(self):
        """Validate that initial value is within specified domain."""
        if self.initial_value is not None and self.domain is not None:
            if not (self.domain[0] <= self.initial_value <= self.domain[1]):
                raise ValueError(
                    f"Initial value {self.initial_value} is outside domain [{self.domain[0]}, {self.domain[1]}]"
                )
        return self

    @field_validator('dependencies')
    @classmethod
    def validate_dependencies_list(cls, v):
        """Validate dependencies are properly formatted."""
        if not isinstance(v, list):
            raise ValueError("Dependencies must be a list")

        # Ensure all dependencies are strings (mathematical symbols)
        for dep in v:
            if not isinstance(dep, str):
                raise ValueError("All dependencies must be mathematical symbol strings")

        return v

    @field_validator('equation_refs')
    @classmethod
    def validate_equation_references(cls, v):
        """Validate equation references are properly formatted."""
        if not isinstance(v, list):
            raise ValueError("Equation references must be a list")

        # Ensure all references are strings
        for ref in v:
            if not isinstance(ref, str):
                raise ValueError("All equation references must be strings")

        return v

    def __str__(self) -> str:
        """String representation preserving mathematical symbol."""
        return f"MathematicalVariable({self.symbol}: {self.description} [{self.variable_type}])"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"MathematicalVariable("
            f"symbol='{self.symbol}', "
            f"description='{self.description}', "
            f"units='{self.units}', "
            f"variable_type='{self.variable_type}', "
            f"domain={self.domain}, "
            f"initial_value={self.initial_value}"
            f")"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on mathematical symbol and properties."""
        if not isinstance(other, MathematicalVariable):
            return False

        return (
            self.symbol == other.symbol and
            self.description == other.description and
            self.units == other.units and
            self.variable_type == other.variable_type and
            self.domain == other.domain and
            self.initial_value == other.initial_value and
            self.dependencies == other.dependencies and
            self.equation_refs == other.equation_refs
        )

    def __hash__(self) -> int:
        """Hash based on mathematical symbol for use in sets and dicts."""
        return hash((
            self.symbol,
            self.description,
            self.units,
            self.variable_type,
            tuple(self.domain) if self.domain else None,
            self.initial_value,
            tuple(self.dependencies),
            tuple(self.equation_refs)
        ))

    def get_all_symbols(self) -> List[str]:
        """Get all mathematical symbols associated with this variable."""
        symbols = [self.symbol]
        symbols.extend(self.dependencies)
        return symbols

    def is_dependent_on(self, other_symbol: str) -> bool:
        """Check if this variable depends on another symbol."""
        return other_symbol in self.dependencies

    def add_dependency(self, symbol: str) -> None:
        """Add a dependency to another mathematical variable."""
        if symbol not in self.dependencies:
            self.dependencies.append(symbol)

    def add_equation_reference(self, equation_id: str) -> None:
        """Add a reference to an equation containing this variable."""
        if equation_id not in self.equation_refs:
            self.equation_refs.append(equation_id)

    def is_in_domain(self, value: float) -> bool:
        """Check if a value is within the variable's domain."""
        if self.domain is None:
            return True
        return self.domain[0] <= value <= self.domain[1]

    def validate_mathematical_consistency(self) -> bool:
        """Validate mathematical consistency of the variable definition."""
        # Check initial value is in domain
        if self.initial_value is not None and self.domain is not None:
            if not self.is_in_domain(self.initial_value):
                return False

        # Check for circular dependencies (basic check)
        if self.symbol in self.dependencies:
            return False

        return True