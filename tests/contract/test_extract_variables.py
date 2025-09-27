"""Contract test for POST /extract/variables endpoint.

This test validates the API contract defined in model_extraction_api.yaml
and MUST FAIL until the endpoint is implemented.
"""

import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any


@pytest.fixture
def sample_latex_content() -> str:
    """Sample LaTeX content with mathematical expressions for testing."""
    return r"""
    \begin{equation}
    dP_{F,t} = \kappa_F((1+\mu_F)C_{P,t} - P_{F,t})dt + \sigma_F P_{F,t}dW_{F,t} + J_F P_{F,t-}dN_t
    \end{equation}

    where $P_{F,t}$ is the producer price at time $t$, $\kappa_F$ is the price adjustment speed,
    $\mu_F$ is the producer profit margin, and $\sigma_F$ is the price volatility parameter.
    """


@pytest.fixture
def extraction_request_payload(sample_latex_content: str) -> Dict[str, Any]:
    """Standard request payload for variable extraction."""
    return {
        "paper_content": sample_latex_content,
        "extraction_options": {
            "include_units": True,
            "validate_dimensions": True,
            "anti_hallucination": True
        }
    }


class TestExtractVariablesContract:
    """Contract tests for POST /extract/variables endpoint."""

    def test_extract_variables_endpoint_exists(self, client: TestClient):
        """Test that the extract variables endpoint exists and returns expected structure."""
        # This test MUST FAIL until the endpoint is implemented
        with pytest.raises(Exception, match="404|not found|endpoint.*not.*implemented"):
            response = client.post("/extract/variables", json={})

    def test_extract_variables_request_schema_validation(
        self, client: TestClient, extraction_request_payload: Dict[str, Any]
    ):
        """Test request schema validation according to OpenAPI specification."""
        # Test valid request structure
        response = client.post("/extract/variables", json=extraction_request_payload)

        # Should fail with implementation error, not schema validation error
        assert response.status_code != 422, "Request schema validation failed"

    def test_extract_variables_response_schema_200(
        self, client: TestClient, extraction_request_payload: Dict[str, Any]
    ):
        """Test successful response schema matches OpenAPI specification."""
        response = client.post("/extract/variables", json=extraction_request_payload)

        if response.status_code == 200:
            data = response.json()

            # Validate response structure according to OpenAPI schema
            assert "variables" in data, "Response missing 'variables' field"
            assert "extraction_report" in data, "Response missing 'extraction_report' field"
            assert isinstance(data["variables"], list), "Variables field must be a list"

            # Validate MathematicalVariable schema for each variable
            for variable in data["variables"]:
                self._validate_mathematical_variable_schema(variable)

            # Validate ExtractionReport schema
            self._validate_extraction_report_schema(data["extraction_report"])

    def test_extract_variables_response_schema_400(
        self, client: TestClient
    ):
        """Test error response schema for invalid input."""
        invalid_payload = {"invalid_field": "invalid_value"}
        response = client.post("/extract/variables", json=invalid_payload)

        if response.status_code == 400:
            data = response.json()
            self._validate_error_schema(data)

    def test_extract_variables_mathematical_fidelity(
        self, client: TestClient, extraction_request_payload: Dict[str, Any]
    ):
        """Test that extracted variables maintain mathematical fidelity."""
        response = client.post("/extract/variables", json=extraction_request_payload)

        if response.status_code == 200:
            variables = response.json()["variables"]

            # Expected variables from the sample LaTeX content
            expected_symbols = {"P_{F,t}", "\\kappa_F", "\\mu_F", "C_{P,t}", "\\sigma_F", "J_F", "W_{F,t}", "N_t"}
            extracted_symbols = {var["symbol"] for var in variables}

            # Verify no mathematical symbols are missing or hallucinated
            assert expected_symbols.issubset(extracted_symbols), \
                f"Missing expected symbols: {expected_symbols - extracted_symbols}"

    def test_extract_variables_anti_hallucination_validation(
        self, client: TestClient, extraction_request_payload: Dict[str, Any]
    ):
        """Test anti-hallucination validation is enforced."""
        response = client.post("/extract/variables", json=extraction_request_payload)

        if response.status_code == 200:
            extraction_report = response.json()["extraction_report"]

            # Verify anti-hallucination checks were performed
            assert "anti_hallucination_flags" in extraction_report
            assert isinstance(extraction_report["anti_hallucination_flags"], list)

    def _validate_mathematical_variable_schema(self, variable: Dict[str, Any]) -> None:
        """Validate MathematicalVariable schema according to OpenAPI specification."""
        required_fields = ["symbol", "description", "units", "variable_type"]
        for field in required_fields:
            assert field in variable, f"MathematicalVariable missing required field: {field}"

        # Validate field types
        assert isinstance(variable["symbol"], str), "Symbol must be string"
        assert isinstance(variable["description"], str), "Description must be string"
        assert isinstance(variable["units"], str), "Units must be string"
        assert variable["variable_type"] in ["STATE", "ALGEBRAIC", "PARAMETER", "CONSTANT"], \
            f"Invalid variable_type: {variable['variable_type']}"

        # Validate optional fields if present
        if "domain" in variable:
            assert isinstance(variable["domain"], list), "Domain must be list"
            assert len(variable["domain"]) == 2, "Domain must have exactly 2 elements"

        if "dependencies" in variable:
            assert isinstance(variable["dependencies"], list), "Dependencies must be list"

        if "equation_refs" in variable:
            assert isinstance(variable["equation_refs"], list), "Equation refs must be list"

    def _validate_extraction_report_schema(self, report: Dict[str, Any]) -> None:
        """Validate ExtractionReport schema according to OpenAPI specification."""
        required_fields = ["extraction_timestamp", "source_document", "extracted_components",
                          "validation_checks", "anti_hallucination_flags"]

        for field in required_fields:
            assert field in report, f"ExtractionReport missing required field: {field}"

        assert isinstance(report["validation_checks"], list), "Validation checks must be list"
        assert isinstance(report["anti_hallucination_flags"], list), "Anti-hallucination flags must be list"

    def _validate_error_schema(self, error: Dict[str, Any]) -> None:
        """Validate Error schema according to OpenAPI specification."""
        required_fields = ["error_code", "message"]
        for field in required_fields:
            assert field in error, f"Error schema missing required field: {field}"

        assert isinstance(error["error_code"], str), "Error code must be string"
        assert isinstance(error["message"], str), "Error message must be string"


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client - will be connected to actual app when implemented."""
    # This fixture MUST FAIL until the FastAPI app is implemented
    from src.api.main import app  # This import will fail initially
    return TestClient(app)