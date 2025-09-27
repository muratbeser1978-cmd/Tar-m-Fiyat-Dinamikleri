"""Unit tests for MathematicalVariable model.

This test MUST FAIL until the MathematicalVariable model is implemented.
Tests validate Pydantic model structure, validation rules, and mathematical constraints.
"""

import pytest
from typing import List, Optional, Any, Union
from pydantic import ValidationError


class TestMathematicalVariableModel:
    """Unit tests for MathematicalVariable Pydantic model."""

    def test_mathematical_variable_import_succeeds(self):
        """Test that MathematicalVariable import succeeds after implementation."""
        from src.core.variables.mathematical_variable import MathematicalVariable
        assert MathematicalVariable is not None

    def test_mathematical_variable_basic_creation(self):
        """Test basic creation of MathematicalVariable with required fields."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Test valid variable creation
        variable = MathematicalVariable(
            symbol="P_{F,t}",
            description="Producer price at time t",
            units="currency/unit",
            variable_type="STATE"
        )

        assert variable.symbol == "P_{F,t}"
        assert variable.description == "Producer price at time t"
        assert variable.units == "currency/unit"
        assert variable.variable_type == "STATE"

    def test_mathematical_variable_symbol_validation(self):
        """Test symbol field validation rules."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Valid symbols
        valid_symbols = ["x", "P_{F,t}", "\\kappa_F", "\\sigma^2", "X_t", "μ", "θ_1"]
        for symbol in valid_symbols:
            variable = MathematicalVariable(
                symbol=symbol,
                description="Test variable",
                units="dimensionless",
                variable_type="PARAMETER"
            )
            assert variable.symbol == symbol

        # Invalid symbols (empty or None)
        with pytest.raises(ValidationError):
            MathematicalVariable(
                symbol="",
                description="Test variable",
                units="dimensionless",
                variable_type="PARAMETER"
            )

        with pytest.raises(ValidationError):
            MathematicalVariable(
                symbol=None,
                description="Test variable",
                units="dimensionless",
                variable_type="PARAMETER"
            )

    def test_variable_type_validation(self):
        """Test variable_type field validation with enum constraints."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Valid variable types
        valid_types = ["STATE", "ALGEBRAIC", "PARAMETER", "CONSTANT"]
        for var_type in valid_types:
            variable = MathematicalVariable(
                symbol="x",
                description="Test variable",
                units="dimensionless",
                variable_type=var_type
            )
            assert variable.variable_type == var_type

        # Invalid variable type
        with pytest.raises(ValidationError):
            MathematicalVariable(
                symbol="x",
                description="Test variable",
                units="dimensionless",
                variable_type="INVALID_TYPE"
            )

    def test_domain_validation(self):
        """Test domain field validation for mathematical constraints."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Valid domains
        valid_domains = [
            [0, 1],                    # Bounded interval
            [0, float('inf')],         # Semi-infinite
            [-float('inf'), float('inf')],  # Real line
            [-1, 1],                   # Symmetric interval
            [0.001, 1000.0]           # Positive range
        ]

        for domain in valid_domains:
            variable = MathematicalVariable(
                symbol="x",
                description="Test variable",
                units="dimensionless",
                variable_type="STATE",
                domain=domain
            )
            assert variable.domain == domain

        # Invalid domains
        invalid_domains = [
            [1],           # Too few elements
            [0, 1, 2],     # Too many elements
            [1, 0],        # Invalid order (max < min)
            ["a", "b"],    # Non-numeric
        ]

        for domain in invalid_domains:
            with pytest.raises(ValidationError):
                MathematicalVariable(
                    symbol="x",
                    description="Test variable",
                    units="dimensionless",
                    variable_type="STATE",
                    domain=domain
                )

    def test_units_validation_and_normalization(self):
        """Test units field validation and normalization."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Valid units
        valid_units = [
            "dimensionless",
            "currency/unit",
            "1/time",
            "kg*m/s^2",
            "m^2/s",
            "USD",
            "percentage"
        ]

        for units in valid_units:
            variable = MathematicalVariable(
                symbol="x",
                description="Test variable",
                units=units,
                variable_type="PARAMETER"
            )
            assert variable.units == units

        # Empty units should not be allowed
        with pytest.raises(ValidationError):
            MathematicalVariable(
                symbol="x",
                description="Test variable",
                units="",
                variable_type="PARAMETER"
            )

    def test_initial_value_validation(self):
        """Test initial_value field validation with domain constraints."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Valid initial value within domain
        variable = MathematicalVariable(
            symbol="P",
            description="Price",
            units="currency",
            variable_type="STATE",
            domain=[0, 1000],
            initial_value=100.0
        )
        assert variable.initial_value == 100.0

        # Initial value outside domain should be caught by validation
        with pytest.raises(ValidationError):
            MathematicalVariable(
                symbol="P",
                description="Price",
                units="currency",
                variable_type="STATE",
                domain=[0, 100],
                initial_value=200.0  # Outside domain
            )

        # None initial value should be allowed
        variable = MathematicalVariable(
            symbol="P",
            description="Price",
            units="currency",
            variable_type="STATE",
            initial_value=None
        )
        assert variable.initial_value is None

    def test_dependencies_validation(self):
        """Test dependencies field validation."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Valid dependencies
        variable = MathematicalVariable(
            symbol="y",
            description="Dependent variable",
            units="dimensionless",
            variable_type="ALGEBRAIC",
            dependencies=["x", "z", "\\theta"]
        )
        assert variable.dependencies == ["x", "z", "\\theta"]

        # Empty dependencies list should be allowed
        variable = MathematicalVariable(
            symbol="x",
            description="Independent variable",
            units="dimensionless",
            variable_type="STATE",
            dependencies=[]
        )
        assert variable.dependencies == []

        # Dependencies should be list of strings
        with pytest.raises(ValidationError):
            MathematicalVariable(
                symbol="y",
                description="Dependent variable",
                units="dimensionless",
                variable_type="ALGEBRAIC",
                dependencies="x"  # String instead of list
            )

    def test_equation_refs_validation(self):
        """Test equation_refs field validation."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Valid equation references
        variable = MathematicalVariable(
            symbol="P_{F,t}",
            description="Producer price",
            units="currency",
            variable_type="STATE",
            equation_refs=["eq_1", "eq_sde_main", "eq_constraint_1"]
        )
        assert variable.equation_refs == ["eq_1", "eq_sde_main", "eq_constraint_1"]

        # Empty equation refs should be allowed
        variable = MathematicalVariable(
            symbol="P",
            description="Price",
            units="currency",
            variable_type="STATE",
            equation_refs=[]
        )
        assert variable.equation_refs == []

    def test_mathematical_variable_json_serialization(self):
        """Test JSON serialization and deserialization."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        original = MathematicalVariable(
            symbol="P_{F,t}",
            description="Producer price at time t",
            units="currency/unit",
            variable_type="STATE",
            domain=[0, float('inf')],
            initial_value=100.0,
            dependencies=["C_{P,t}"],
            equation_refs=["eq_1"]
        )

        # Test serialization
        json_data = original.model_dump()
        assert json_data["symbol"] == "P_{F,t}"
        assert json_data["variable_type"] == "STATE"

        # Test deserialization
        restored = MathematicalVariable.model_validate(json_data)
        assert restored.symbol == original.symbol
        assert restored.description == original.description
        assert restored.units == original.units
        assert restored.variable_type == original.variable_type
        assert restored.domain == original.domain
        assert restored.initial_value == original.initial_value
        assert restored.dependencies == original.dependencies
        assert restored.equation_refs == original.equation_refs

    def test_mathematical_variable_string_representation(self):
        """Test string representation of MathematicalVariable."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        variable = MathematicalVariable(
            symbol="P_{F,t}",
            description="Producer price at time t",
            units="currency/unit",
            variable_type="STATE"
        )

        str_repr = str(variable)
        assert "P_{F,t}" in str_repr
        assert "Producer price at time t" in str_repr
        assert "STATE" in str_repr

    def test_mathematical_variable_equality(self):
        """Test equality comparison between MathematicalVariable instances."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        var1 = MathematicalVariable(
            symbol="x",
            description="Test variable",
            units="dimensionless",
            variable_type="PARAMETER"
        )

        var2 = MathematicalVariable(
            symbol="x",
            description="Test variable",
            units="dimensionless",
            variable_type="PARAMETER"
        )

        var3 = MathematicalVariable(
            symbol="y",
            description="Different variable",
            units="dimensionless",
            variable_type="PARAMETER"
        )

        assert var1 == var2, "Variables with same attributes should be equal"
        assert var1 != var3, "Variables with different attributes should not be equal"

    def test_mathematical_variable_hash(self):
        """Test hash functionality for use in sets and dictionaries."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        var1 = MathematicalVariable(
            symbol="x",
            description="Test variable",
            units="dimensionless",
            variable_type="PARAMETER"
        )

        var2 = MathematicalVariable(
            symbol="x",
            description="Test variable",
            units="dimensionless",
            variable_type="PARAMETER"
        )

        # Should be usable in sets
        var_set = {var1, var2}
        assert len(var_set) == 1, "Equal variables should have same hash"

        # Should be usable as dictionary keys
        var_dict = {var1: "value1"}
        assert var_dict[var2] == "value1", "Equal variables should work as dict keys"

    def test_mathematical_fidelity_preservation(self):
        """Test that mathematical symbols are preserved exactly without modification."""
        try:
            from src.core.variables.mathematical_variable import MathematicalVariable
        except ImportError:
            pytest.skip("MathematicalVariable not implemented yet")

        # Complex mathematical symbols should be preserved exactly
        complex_symbols = [
            "\\partial P_{F,t}/\\partial t",
            "\\mathbb{E}[\\Delta P]",
            "\\int_0^T f(s)ds",
            "\\sum_{i=1}^n \\alpha_i x_i",
            "P_{F,t}^{(n)}",
            "\\tilde{P}_{F,t}"
        ]

        for symbol in complex_symbols:
            variable = MathematicalVariable(
                symbol=symbol,
                description=f"Mathematical symbol: {symbol}",
                units="dimensionless",
                variable_type="PARAMETER"
            )
            # Symbol must be preserved exactly
            assert variable.symbol == symbol, f"Symbol {symbol} was modified"


@pytest.fixture
def sample_variable_data():
    """Sample data for creating MathematicalVariable instances."""
    return {
        "symbol": "P_{F,t}",
        "description": "Producer price at time t",
        "units": "currency/unit",
        "variable_type": "STATE",
        "domain": [0, 1000],
        "initial_value": 100.0,
        "dependencies": ["C_{P,t}", "\\kappa_F"],
        "equation_refs": ["eq_food_price_sde"]
    }