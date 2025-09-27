"""Unit tests for ModelParameter model.

This test MUST FAIL until the ModelParameter model is implemented.
Tests validate Pydantic model with economic constraints and calibration data.
"""

import pytest
from typing import Dict, Any, Optional, List
from pydantic import ValidationError


class TestModelParameterModel:
    """Unit tests for ModelParameter Pydantic model."""

    def test_model_parameter_import_succeeds(self):
        """Test that ModelParameter import succeeds after implementation."""
        from src.core.variables.model_parameter import ModelParameter
        assert ModelParameter is not None

    def test_model_parameter_basic_creation(self):
        """Test basic creation of ModelParameter with required fields."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        parameter = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed parameter",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL"
        )

        assert parameter.name == "kappa_F"
        assert parameter.symbol == "\\kappa_F"
        assert parameter.description == "Price adjustment speed parameter"
        assert parameter.units == "1/time"
        assert parameter.default_value == 2.0
        assert parameter.parameter_type == "STRUCTURAL"

    def test_parameter_type_validation(self):
        """Test parameter_type field validation with economic categories."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        # Valid parameter types for economic models
        valid_types = [
            "STRUCTURAL",      # Structural economic parameters
            "BEHAVIORAL",      # Behavioral parameters
            "TECHNOLOGICAL",   # Technology parameters
            "POLICY",          # Policy parameters
            "STOCHASTIC",      # Noise/volatility parameters
            "CALIBRATED"       # Externally calibrated parameters
        ]

        for param_type in valid_types:
            parameter = ModelParameter(
                name="test_param",
                symbol="\\theta",
                description="Test parameter",
                units="dimensionless",
                default_value=1.0,
                parameter_type=param_type
            )
            assert parameter.parameter_type == param_type

        # Invalid parameter type
        with pytest.raises(ValidationError):
            ModelParameter(
                name="test_param",
                symbol="\\theta",
                description="Test parameter",
                units="dimensionless",
                default_value=1.0,
                parameter_type="INVALID_TYPE"
            )

    def test_value_range_validation(self):
        """Test value_range field validation for economic constraints."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        # Valid value ranges
        valid_ranges = [
            [0, 1],                    # Unit interval (e.g., elasticities)
            [0, float('inf')],         # Non-negative (e.g., prices, quantities)
            [-1, 1],                   # Symmetric interval (e.g., correlations)
            [0.01, 10.0],             # Positive bounded (e.g., adjustment speeds)
            [-float('inf'), float('inf')]  # Unrestricted
        ]

        for value_range in valid_ranges:
            parameter = ModelParameter(
                name="test_param",
                symbol="\\theta",
                description="Test parameter",
                units="dimensionless",
                default_value=0.5,
                parameter_type="STRUCTURAL",
                value_range=value_range
            )
            assert parameter.value_range == value_range

        # Invalid ranges
        invalid_ranges = [
            [1, 0],        # max < min
            [0],           # Too few elements
            [0, 1, 2],     # Too many elements
        ]

        for value_range in invalid_ranges:
            with pytest.raises(ValidationError):
                ModelParameter(
                    name="test_param",
                    symbol="\\theta",
                    description="Test parameter",
                    units="dimensionless",
                    default_value=0.5,
                    parameter_type="STRUCTURAL",
                    value_range=value_range
                )

    def test_default_value_within_range_validation(self):
        """Test that default_value is validated against value_range."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        # Valid default value within range
        parameter = ModelParameter(
            name="elasticity",
            symbol="\\epsilon",
            description="Price elasticity",
            units="dimensionless",
            default_value=0.5,
            parameter_type="STRUCTURAL",
            value_range=[0, 1]
        )
        assert parameter.default_value == 0.5

        # Default value outside range should be rejected
        with pytest.raises(ValidationError):
            ModelParameter(
                name="elasticity",
                symbol="\\epsilon",
                description="Price elasticity",
                units="dimensionless",
                default_value=1.5,  # Outside range [0, 1]
                parameter_type="STRUCTURAL",
                value_range=[0, 1]
            )

    def test_economic_interpretation_validation(self):
        """Test economic_interpretation field for model documentation."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        interpretation = {
            "economic_meaning": "Speed of price adjustment to equilibrium",
            "typical_values": "1.0 to 5.0 for agricultural markets",
            "estimation_method": "Generalized Method of Moments",
            "sensitivity": "High impact on price volatility",
            "policy_relevance": "Market intervention effectiveness"
        }

        parameter = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL",
            economic_interpretation=interpretation
        )

        assert parameter.economic_interpretation == interpretation
        assert parameter.economic_interpretation["economic_meaning"] == interpretation["economic_meaning"]

    def test_calibration_info_validation(self):
        """Test calibration_info field for parameter estimation metadata."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        calibration_info = {
            "data_source": "FAO food price database 1990-2020",
            "estimation_period": "2010-2020",
            "method": "Maximum Likelihood Estimation",
            "standard_error": 0.15,
            "confidence_interval": [1.7, 2.3],
            "t_statistic": 13.33,
            "p_value": 0.001
        }

        parameter = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed",
            units="1/time",
            default_value=2.0,
            parameter_type="CALIBRATED",
            calibration_info=calibration_info
        )

        assert parameter.calibration_info == calibration_info
        assert parameter.calibration_info["standard_error"] == 0.15

    def test_prior_distribution_validation(self):
        """Test prior_distribution field for Bayesian estimation."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        prior_distribution = {
            "distribution_type": "normal",
            "parameters": {
                "mean": 2.0,
                "variance": 0.25
            },
            "support": [0, float('inf')],
            "rationale": "Based on literature review of similar agricultural markets"
        }

        parameter = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL",
            prior_distribution=prior_distribution
        )

        assert parameter.prior_distribution == prior_distribution
        assert parameter.prior_distribution["distribution_type"] == "normal"

    def test_related_parameters_validation(self):
        """Test related_parameters field for parameter dependencies."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        related_params = [
            {
                "parameter": "sigma_F",
                "relationship": "volatility_scaling",
                "constraint": "sigma_F >= 0.1 * kappa_F"
            },
            {
                "parameter": "mu_F",
                "relationship": "profit_margin",
                "constraint": "mu_F > 0 when kappa_F > 1"
            }
        ]

        parameter = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL",
            related_parameters=related_params
        )

        assert parameter.related_parameters == related_params
        assert len(parameter.related_parameters) == 2

    def test_model_parameter_json_serialization(self):
        """Test JSON serialization and deserialization."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        original = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed parameter",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL",
            value_range=[0, 10],
            economic_interpretation={
                "economic_meaning": "Speed of price adjustment"
            },
            calibration_info={
                "method": "MLE",
                "standard_error": 0.15
            }
        )

        # Test serialization
        json_data = original.model_dump()
        assert json_data["name"] == "kappa_F"
        assert json_data["parameter_type"] == "STRUCTURAL"

        # Test deserialization
        restored = ModelParameter.model_validate(json_data)
        assert restored.name == original.name
        assert restored.symbol == original.symbol
        assert restored.default_value == original.default_value
        assert restored.parameter_type == original.parameter_type
        assert restored.economic_interpretation == original.economic_interpretation

    def test_parameter_validation_with_constraints(self):
        """Test parameter validation with economic constraints."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        # Test constraint validation
        constraints = [
            "kappa_F > 0",  # Positivity constraint
            "kappa_F < lambda_max",  # Upper bound constraint
            "kappa_F * tau < 1"  # Stability constraint
        ]

        parameter = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL",
            constraints=constraints
        )

        assert parameter.constraints == constraints

    def test_sensitivity_analysis_info(self):
        """Test sensitivity_analysis field for parameter importance."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        sensitivity_info = {
            "local_sensitivity": {
                "first_order": 0.15,
                "second_order": 0.03
            },
            "global_sensitivity": {
                "sobol_first": 0.22,
                "sobol_total": 0.31
            },
            "elasticity": {
                "price_elasticity": 0.45,
                "quantity_elasticity": -0.30
            }
        }

        parameter = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL",
            sensitivity_analysis=sensitivity_info
        )

        assert parameter.sensitivity_analysis == sensitivity_info
        assert parameter.sensitivity_analysis["local_sensitivity"]["first_order"] == 0.15

    def test_model_parameter_inequality_comparison(self):
        """Test inequality comparisons for parameter ordering."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        param1 = ModelParameter(
            name="kappa_1",
            symbol="\\kappa_1",
            description="Parameter 1",
            units="1/time",
            default_value=1.0,
            parameter_type="STRUCTURAL"
        )

        param2 = ModelParameter(
            name="kappa_2",
            symbol="\\kappa_2",
            description="Parameter 2",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL"
        )

        # Should be able to compare by default_value
        assert param1 < param2
        assert param2 > param1
        assert param1 != param2

    def test_parameter_copy_with_update(self):
        """Test parameter copying with value updates."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        original = ModelParameter(
            name="kappa_F",
            symbol="\\kappa_F",
            description="Price adjustment speed",
            units="1/time",
            default_value=2.0,
            parameter_type="STRUCTURAL"
        )

        # Create copy with updated value
        updated = original.model_copy(update={"default_value": 3.0})

        assert original.default_value == 2.0  # Original unchanged
        assert updated.default_value == 3.0   # Copy updated
        assert updated.name == original.name   # Other fields preserved

    def test_mathematical_fidelity_preservation(self):
        """Test that mathematical symbols and expressions are preserved exactly."""
        try:
            from src.core.variables.model_parameter import ModelParameter
        except ImportError:
            pytest.skip("ModelParameter not implemented yet")

        complex_symbols = [
            "\\kappa_{F,adjusted}",
            "\\sigma_F^{(2)}",
            "\\mathbb{E}[\\epsilon_t]",
            "\\partial \\pi / \\partial p"
        ]

        for symbol in complex_symbols:
            parameter = ModelParameter(
                name=f"param_{symbol}",
                symbol=symbol,
                description=f"Parameter with symbol {symbol}",
                units="dimensionless",
                default_value=1.0,
                parameter_type="STRUCTURAL"
            )
            # Symbol must be preserved exactly
            assert parameter.symbol == symbol, f"Symbol {symbol} was modified"


@pytest.fixture
def sample_parameter_data():
    """Sample data for creating ModelParameter instances."""
    return {
        "name": "kappa_F",
        "symbol": "\\kappa_F",
        "description": "Price adjustment speed parameter for food markets",
        "units": "1/time",
        "default_value": 2.0,
        "parameter_type": "STRUCTURAL",
        "value_range": [0, 10],
        "economic_interpretation": {
            "economic_meaning": "Speed of price adjustment to long-run equilibrium",
            "typical_values": "1.0 to 5.0 for agricultural commodity markets"
        },
        "calibration_info": {
            "data_source": "FAO food price database",
            "method": "Maximum Likelihood Estimation",
            "standard_error": 0.15
        }
    }