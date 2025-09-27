"""Mathematical Equation model with LaTeX parsing and exact preservation.

This module implements the MathematicalEquation model with strict mathematical fidelity
requirements, ensuring exact preservation of LaTeX forms and mathematical structure.
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class EquationType(str, Enum):
    """Enumeration of mathematical equation types."""
    SDE = "SDE"                 # Stochastic Differential Equation
    ODE = "ODE"                 # Ordinary Differential Equation
    ALGEBRAIC = "ALGEBRAIC"     # Algebraic equation
    CONSTRAINT = "CONSTRAINT"   # Constraint equation


class MathematicalEquation(BaseModel):
    """Mathematical equation with LaTeX preservation and dependency analysis.

    This model enforces constitutional mathematical fidelity requirements:
    - Exact preservation of LaTeX forms without ANY modification
    - Complete variable and parameter dependency tracking
    - Mathematical structure validation and analysis
    """

    equation_id: str = Field(
        ...,
        description="Unique identifier for the equation",
        min_length=1
    )

    latex_form: str = Field(
        ...,
        description="LaTeX representation preserved exactly",
        min_length=1
    )

    equation_type: EquationType = Field(
        ...,
        description="Type of mathematical equation"
    )

    variables: List[str] = Field(
        ...,
        description="List of variables appearing in the equation"
    )

    parameters: List[str] = Field(
        ...,
        description="List of parameters appearing in the equation"
    )

    source_location: Optional[str] = Field(
        None,
        description="Source location reference (e.g., 'Section 2.3.1, Equation 7')"
    )

    mathematical_properties: Optional[Dict[str, Any]] = Field(
        None,
        description="Mathematical characteristics and properties"
    )

    numerical_scheme: Optional[str] = Field(
        None,
        description="Numerical solution method if applicable"
    )

    stability_conditions: Optional[List[str]] = Field(
        None,
        description="Mathematical stability constraints"
    )

    model_config = ConfigDict(
        # Ensure exact preservation of all fields
        validate_assignment=True,
        # Allow for complex mathematical data structures
        arbitrary_types_allowed=True,
        # Preserve exact LaTeX notation - NO whitespace stripping
        str_strip_whitespace=False
    )

    @field_validator('equation_id')
    @classmethod
    def validate_equation_id(cls, v):
        """Validate equation ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Equation ID cannot be empty")
        return v.strip()

    @field_validator('latex_form')
    @classmethod
    def validate_latex_preservation(cls, v):
        """Validate LaTeX form is preserved exactly - constitutional requirement."""
        if not v:
            raise ValueError("LaTeX form cannot be empty")

        # CONSTITUTIONAL REQUIREMENT: NO MODIFICATION ALLOWED
        # Return exactly as provided to preserve mathematical fidelity
        return v

    @field_validator('variables')
    @classmethod
    def validate_variables_list(cls, v):
        """Validate variables list structure."""
        if not isinstance(v, list):
            raise ValueError("Variables must be a list")

        # Allow empty list for some equation types
        for var in v:
            if not isinstance(var, str):
                raise ValueError("All variables must be strings (mathematical symbols)")

        return v

    @field_validator('parameters')
    @classmethod
    def validate_parameters_list(cls, v):
        """Validate parameters list structure."""
        if not isinstance(v, list):
            raise ValueError("Parameters must be a list")

        # Allow empty list
        for param in v:
            if not isinstance(param, str):
                raise ValueError("All parameters must be strings (mathematical symbols)")

        return v

    @field_validator('mathematical_properties')
    @classmethod
    def validate_mathematical_properties(cls, v):
        """Validate mathematical properties structure."""
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError("Mathematical properties must be a dictionary")

        return v

    @field_validator('stability_conditions')
    @classmethod
    def validate_stability_conditions(cls, v):
        """Validate stability conditions."""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("Stability conditions must be a list")

        for condition in v:
            if not isinstance(condition, str):
                raise ValueError("Each stability condition must be a string")

        return v

    def __str__(self) -> str:
        """String representation preserving equation ID and type."""
        return f"MathematicalEquation({self.equation_id}: {self.equation_type})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"MathematicalEquation("
            f"equation_id='{self.equation_id}', "
            f"equation_type='{self.equation_type}', "
            f"variables={len(self.variables)}, "
            f"parameters={len(self.parameters)}"
            f")"
        )

    def __eq__(self, other) -> bool:
        """Equality comparison based on equation ID and LaTeX form."""
        if not isinstance(other, MathematicalEquation):
            return False

        return (
            self.equation_id == other.equation_id and
            self.latex_form == other.latex_form and
            self.equation_type == other.equation_type and
            self.variables == other.variables and
            self.parameters == other.parameters
        )

    def __hash__(self) -> int:
        """Hash based on equation ID and LaTeX form."""
        return hash((
            self.equation_id,
            self.latex_form,
            self.equation_type,
            tuple(self.variables),
            tuple(self.parameters)
        ))

    def get_all_symbols(self) -> Set[str]:
        """Get all mathematical symbols (variables and parameters) in the equation."""
        return set(self.variables + self.parameters)

    def get_dependent_variables(self) -> List[str]:
        """Get variables that appear on the left-hand side (dependent variables)."""
        # Simple heuristic: look for variables before '=' sign
        if '=' not in self.latex_form:
            return []

        left_side = self.latex_form.split('=')[0]
        dependent_vars = []

        for var in self.variables:
            if var in left_side:
                dependent_vars.append(var)

        return dependent_vars

    def get_independent_variables(self) -> List[str]:
        """Get variables that appear on the right-hand side (independent variables)."""
        if '=' not in self.latex_form:
            return self.variables

        right_side = self.latex_form.split('=', 1)[1]
        independent_vars = []

        for var in self.variables:
            if var in right_side:
                independent_vars.append(var)

        return independent_vars

    def has_stochastic_terms(self) -> bool:
        """Check if equation contains stochastic differential terms."""
        stochastic_indicators = ['dW', 'dB', 'dN', 'Brownian', 'Wiener', 'Poisson']
        return any(indicator in self.latex_form for indicator in stochastic_indicators)

    def has_jump_terms(self) -> bool:
        """Check if equation contains jump terms."""
        jump_indicators = ['dN', 'Poisson', 'Jump', 'N_t', 'N_{']
        return any(indicator in self.latex_form for indicator in jump_indicators)

    def validate_stochastic_structure(self) -> bool:
        """Validate stochastic equation structure."""
        if self.equation_type == EquationType.SDE:
            # SDE must have stochastic terms
            return self.has_stochastic_terms()
        return True

    def get_equation_complexity(self) -> Dict[str, int]:
        """Analyze equation complexity metrics."""
        return {
            "num_variables": len(self.variables),
            "num_parameters": len(self.parameters),
            "latex_length": len(self.latex_form),
            "num_derivatives": self.latex_form.count('\\frac') + self.latex_form.count('d'),
            "num_integrals": self.latex_form.count('\\int'),
            "num_sums": self.latex_form.count('\\sum'),
            "num_products": self.latex_form.count('\\prod')
        }

    def extract_mathematical_operators(self) -> List[str]:
        """Extract mathematical operators present in the equation."""
        operators = []
        operator_patterns = [
            '\\frac', '\\partial', '\\int', '\\sum', '\\prod',
            '\\sin', '\\cos', '\\tan', '\\exp', '\\log', '\\ln',
            '\\sqrt', '\\alpha', '\\beta', '\\gamma', '\\delta',
            '\\theta', '\\phi', '\\psi', '\\omega', '\\mu', '\\sigma'
        ]

        for op in operator_patterns:
            if op in self.latex_form:
                operators.append(op)

        return operators

    def validate_mathematical_consistency(self) -> bool:
        """Validate mathematical consistency of the equation."""
        # Check that all variables and parameters are actually in the LaTeX form
        all_symbols = self.get_all_symbols()

        for symbol in all_symbols:
            if symbol not in self.latex_form:
                return False

        # Check equation type consistency
        if not self.validate_stochastic_structure():
            return False

        return True

    def parse_latex_structure(self) -> Dict[str, Any]:
        """Parse LaTeX structure while preserving exact form."""
        structure = {
            "original_latex": self.latex_form,  # Preserve exactly
            "equation_type": self.equation_type,
            "has_fractions": "\\frac" in self.latex_form,
            "has_derivatives": "\\partial" in self.latex_form or self.latex_form.startswith("d"),
            "has_integrals": "\\int" in self.latex_form,
            "has_summations": "\\sum" in self.latex_form,
            "has_products": "\\prod" in self.latex_form,
            "has_matrices": "\\begin{pmatrix}" in self.latex_form or "\\begin{matrix}" in self.latex_form,
            "has_cases": "\\begin{cases}" in self.latex_form,
            "operators": self.extract_mathematical_operators(),
            "complexity": self.get_equation_complexity()
        }

        return structure

    def get_referenced_equations(self) -> List[str]:
        """Get equation references that this equation depends on."""
        # This would be populated based on dependency analysis
        return []

    def add_mathematical_property(self, property_name: str, property_value: Any) -> None:
        """Add a mathematical property to the equation."""
        if self.mathematical_properties is None:
            self.mathematical_properties = {}

        self.mathematical_properties[property_name] = property_value

    def get_mathematical_property(self, property_name: str) -> Optional[Any]:
        """Get a specific mathematical property."""
        if self.mathematical_properties is None:
            return None

        return self.mathematical_properties.get(property_name)

    def validate_constitutional_compliance(self) -> bool:
        """Validate compliance with constitutional mathematical fidelity requirements."""
        # CONSTITUTIONAL REQUIREMENT: LaTeX form must be preserved exactly
        if not self.latex_form:
            return False

        # CONSTITUTIONAL REQUIREMENT: No mathematical symbols missing
        if not self.validate_mathematical_consistency():
            return False

        # CONSTITUTIONAL REQUIREMENT: Equation structure validation
        if not self.validate_stochastic_structure():
            return False

        return True