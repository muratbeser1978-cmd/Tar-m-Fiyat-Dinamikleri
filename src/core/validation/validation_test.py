"""Validation Test model for mathematical validation framework.

This module implements the ValidationTest model with multiple test types
for comprehensive mathematical model validation.
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class TestType(str, Enum):
    """Enumeration of validation test types."""
    ANALYTICAL = "ANALYTICAL"       # Analytical solution comparison
    BENCHMARK = "BENCHMARK"         # Performance benchmarking
    CONVERGENCE = "CONVERGENCE"     # Convergence rate testing
    STATISTICAL = "STATISTICAL"    # Statistical property testing


class ValidationTest(BaseModel):
    """Mathematical validation test with multiple test types.

    This model enforces constitutional mathematical fidelity requirements:
    - Exact preservation of test specifications and expected results
    - Comprehensive validation against analytical solutions
    - Statistical and convergence testing capabilities
    """

    test_id: str = Field(
        ...,
        description="Unique identifier for the validation test",
        min_length=1
    )

    test_type: TestType = Field(
        ...,
        description="Type of validation test to perform"
    )

    description: str = Field(
        ...,
        description="Description of the validation test",
        min_length=1
    )

    input_conditions: Dict[str, Any] = Field(
        ...,
        description="Input conditions and parameters for the test"
    )

    expected_results: Dict[str, Any] = Field(
        ...,
        description="Expected results or reference values"
    )

    tolerance: float = Field(
        ...,
        description="Numerical tolerance for test validation",
        ge=0
    )

    analytical_solution: Optional[str] = Field(
        None,
        description="Analytical solution expression if applicable"
    )

    benchmark_reference: Optional[str] = Field(
        None,
        description="Reference implementation for benchmarking"
    )

    model_config = ConfigDict(
        # Ensure exact preservation of test specifications
        validate_assignment=True,
        # Allow complex test data structures
        arbitrary_types_allowed=True,
        # Preserve exact mathematical expressions
        str_strip_whitespace=False
    )

    @field_validator('test_id', 'description')
    @classmethod
    def validate_required_strings(cls, v):
        """Validate required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Required string field cannot be empty")
        return v.strip()

    @field_validator('tolerance')
    @classmethod
    def validate_tolerance(cls, v):
        """Validate tolerance is non-negative."""
        if v < 0:
            raise ValueError("Tolerance must be non-negative")
        return v

    @field_validator('input_conditions', 'expected_results')
    @classmethod
    def validate_test_data(cls, v):
        """Validate test data dictionaries."""
        if not isinstance(v, dict):
            raise ValueError("Test data must be a dictionary")
        return v

    def __str__(self) -> str:
        """String representation preserving test ID and type."""
        return f"ValidationTest({self.test_id}: {self.test_type})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"ValidationTest("
            f"test_id='{self.test_id}', "
            f"test_type='{self.test_type}', "
            f"tolerance={self.tolerance}"
            f")"
        )

    def is_analytical_test(self) -> bool:
        """Check if this is an analytical validation test."""
        return self.test_type == TestType.ANALYTICAL

    def is_convergence_test(self) -> bool:
        """Check if this is a convergence rate test."""
        return self.test_type == TestType.CONVERGENCE

    def has_analytical_solution(self) -> bool:
        """Check if analytical solution is provided."""
        return self.analytical_solution is not None

    def get_test_parameters(self) -> Dict[str, Any]:
        """Get all test parameters including conditions and expected results."""
        return {
            "input_conditions": self.input_conditions,
            "expected_results": self.expected_results,
            "tolerance": self.tolerance
        }

    def validate_constitutional_compliance(self) -> bool:
        """Validate constitutional mathematical fidelity requirements."""
        # CONSTITUTIONAL REQUIREMENT: Test specifications preserved exactly
        if not self.test_id or not self.description:
            return False

        # CONSTITUTIONAL REQUIREMENT: Mathematical expressions preserved
        if self.analytical_solution and not isinstance(self.analytical_solution, str):
            return False

        return True