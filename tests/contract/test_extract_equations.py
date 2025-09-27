"""Contract test for POST /extract/equations endpoint.

This test validates the API contract defined in model_extraction_api.yaml
and MUST FAIL until the endpoint is implemented.
"""

import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any


@pytest.fixture
def sample_equation_content() -> str:
    """Sample content with various equation types for testing."""
    return r"""
    \begin{equation}
    \label{eq:food_price_sde}
    dP_{F,t} = \kappa_F((1+\mu_F)C_{P,t} - P_{F,t})dt + \sigma_F P_{F,t}dW_{F,t} + J_F P_{F,t-}dN_t
    \end{equation}

    \begin{equation}
    \label{eq:consumer_price}
    C_{P,t} = \alpha P_{F,t} + \beta
    \end{equation}

    \begin{equation}
    \label{eq:jump_constraint}
    \mathbb{E}[J_F] = 0, \quad \text{Var}(J_F) = \sigma_J^2
    \end{equation}
    """


@pytest.fixture
def equation_extraction_payload(sample_equation_content: str) -> Dict[str, Any]:
    """Standard request payload for equation extraction."""
    return {
        "paper_content": sample_equation_content,
        "equation_types": ["SDE", "ALGEBRAIC", "CONSTRAINT"],
        "preservation_mode": "EXACT"
    }


class TestExtractEquationsContract:
    """Contract tests for POST /extract/equations endpoint."""

    def test_extract_equations_endpoint_exists(self, client: TestClient):
        """Test that the extract equations endpoint exists."""
        with pytest.raises(Exception, match="404|not found|endpoint.*not.*implemented"):
            response = client.post("/extract/equations", json={})

    def test_extract_equations_request_schema(
        self, client: TestClient, equation_extraction_payload: Dict[str, Any]
    ):
        """Test request schema validation for equation extraction."""
        response = client.post("/extract/equations", json=equation_extraction_payload)
        assert response.status_code != 422, "Request schema validation failed"

    def test_extract_equations_response_schema_200(
        self, client: TestClient, equation_extraction_payload: Dict[str, Any]
    ):
        """Test successful response schema for equation extraction."""
        response = client.post("/extract/equations", json=equation_extraction_payload)

        if response.status_code == 200:
            data = response.json()

            # Validate response structure
            assert "equations" in data, "Response missing 'equations' field"
            assert "dependency_graph" in data, "Response missing 'dependency_graph' field"
            assert isinstance(data["equations"], list), "Equations field must be a list"

            # Validate each equation schema
            for equation in data["equations"]:
                self._validate_mathematical_equation_schema(equation)

            # Validate dependency graph schema
            self._validate_dependency_graph_schema(data["dependency_graph"])

    def test_exact_equation_preservation(
        self, client: TestClient, equation_extraction_payload: Dict[str, Any]
    ):
        """Test that equations are preserved exactly without modification."""
        response = client.post("/extract/equations", json=equation_extraction_payload)

        if response.status_code == 200:
            equations = response.json()["equations"]

            # Expected LaTeX forms from sample content
            expected_latex_forms = {
                r"dP_{F,t} = \kappa_F((1+\mu_F)C_{P,t} - P_{F,t})dt + \sigma_F P_{F,t}dW_{F,t} + J_F P_{F,t-}dN_t",
                r"C_{P,t} = \alpha P_{F,t} + \beta",
                r"\mathbb{E}[J_F] = 0, \quad \text{Var}(J_F) = \sigma_J^2"
            }

            extracted_latex = {eq["latex_form"] for eq in equations}

            # Verify exact preservation (no simplification or modification)
            for expected in expected_latex_forms:
                assert any(expected in latex for latex in extracted_latex), \
                    f"Expected LaTeX form not found or modified: {expected}"

    def test_equation_type_classification(
        self, client: TestClient, equation_extraction_payload: Dict[str, Any]
    ):
        """Test correct classification of equation types."""
        response = client.post("/extract/equations", json=equation_extraction_payload)

        if response.status_code == 200:
            equations = response.json()["equations"]
            equation_types = {eq["equation_type"] for eq in equations}

            # Should include the types from our sample content
            expected_types = {"SDE", "ALGEBRAIC", "CONSTRAINT"}
            assert expected_types.issubset(equation_types), \
                f"Missing expected equation types: {expected_types - equation_types}"

    def test_dependency_graph_completeness(
        self, client: TestClient, equation_extraction_payload: Dict[str, Any]
    ):
        """Test that dependency graph captures all variable relationships."""
        response = client.post("/extract/equations", json=equation_extraction_payload)

        if response.status_code == 200:
            dependency_graph = response.json()["dependency_graph"]

            # Verify graph has nodes for all variables
            nodes = set(dependency_graph["nodes"])
            expected_variables = {"P_{F,t}", "C_{P,t}", "\\kappa_F", "\\mu_F", "\\alpha", "\\beta", "J_F"}

            assert expected_variables.issubset(nodes), \
                f"Dependency graph missing variables: {expected_variables - nodes}"

            # Verify evaluation order is provided
            assert "evaluation_order" in dependency_graph
            assert isinstance(dependency_graph["evaluation_order"], list)

    def _validate_mathematical_equation_schema(self, equation: Dict[str, Any]) -> None:
        """Validate MathematicalEquation schema according to OpenAPI specification."""
        required_fields = ["equation_id", "latex_form", "equation_type", "variables", "parameters"]
        for field in required_fields:
            assert field in equation, f"MathematicalEquation missing required field: {field}"

        # Validate field types
        assert isinstance(equation["equation_id"], str), "Equation ID must be string"
        assert isinstance(equation["latex_form"], str), "LaTeX form must be string"
        assert equation["equation_type"] in ["SDE", "ODE", "ALGEBRAIC", "CONSTRAINT"], \
            f"Invalid equation_type: {equation['equation_type']}"
        assert isinstance(equation["variables"], list), "Variables must be list"
        assert isinstance(equation["parameters"], list), "Parameters must be list"

        # Validate optional fields
        if "source_location" in equation:
            assert isinstance(equation["source_location"], str), "Source location must be string"

        if "stability_conditions" in equation:
            assert isinstance(equation["stability_conditions"], list), "Stability conditions must be list"

    def _validate_dependency_graph_schema(self, graph: Dict[str, Any]) -> None:
        """Validate DependencyGraph schema according to OpenAPI specification."""
        required_fields = ["nodes", "edges", "evaluation_order"]
        for field in required_fields:
            assert field in graph, f"DependencyGraph missing required field: {field}"

        assert isinstance(graph["nodes"], list), "Nodes must be list"
        assert isinstance(graph["edges"], list), "Edges must be list"
        assert isinstance(graph["evaluation_order"], list), "Evaluation order must be list"

        # Validate edge structure
        for edge in graph["edges"]:
            assert isinstance(edge, list), "Each edge must be a list"
            assert len(edge) == 2, "Each edge must have exactly 2 elements"

        if "cycles" in graph:
            assert isinstance(graph["cycles"], list), "Cycles must be list"


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client - will be connected to actual app when implemented."""
    from src.api.main import app  # This import will fail initially
    return TestClient(app)