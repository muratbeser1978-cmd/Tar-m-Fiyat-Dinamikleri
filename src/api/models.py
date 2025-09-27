"""Pydantic request and response models for the Mathematical Extraction API.

This module implements comprehensive request/response models with constitutional
mathematical fidelity requirements for API data validation.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from datetime import datetime


# ===== ENUMS =====

class ProcessingMode(str, Enum):
    """Processing mode for extraction."""
    FAST = "FAST"           # Quick extraction with minimal analysis
    STANDARD = "STANDARD"   # Standard extraction with full analysis
    COMPREHENSIVE = "COMPREHENSIVE"  # Deep analysis with all features


class OutputFormat(str, Enum):
    """Output format for extracted content."""
    JSON = "JSON"
    LATEX = "LATEX"
    MARKDOWN = "MARKDOWN"
    XML = "XML"


class ValidationLevel(str, Enum):
    """Level of mathematical validation."""
    BASIC = "BASIC"
    STANDARD = "STANDARD"
    STRICT = "STRICT"


# ===== BASE MODELS =====

class APIResponse(BaseModel):
    """Base response model for all API endpoints."""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message or error description")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    processing_time: float = Field(..., description="Processing time in seconds", ge=0)

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True
    )


class PaginationInfo(BaseModel):
    """Pagination information for list responses."""
    page: int = Field(..., description="Current page number", ge=1)
    per_page: int = Field(..., description="Items per page", ge=1, le=1000)
    total_items: int = Field(..., description="Total number of items", ge=0)
    total_pages: int = Field(..., description="Total number of pages", ge=0)


# ===== EQUATION EXTRACTION MODELS =====

class ExtractEquationsRequest(BaseModel):
    """Request model for equation extraction endpoint."""
    document_content: str = Field(
        ...,
        description="LaTeX document content to process",
        min_length=1,
        max_length=1_000_000
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document identifier"
    )
    processing_mode: ProcessingMode = Field(
        default=ProcessingMode.STANDARD,
        description="Processing mode for extraction"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Desired output format"
    )
    min_confidence: float = Field(
        default=0.5,
        description="Minimum confidence threshold for equations",
        ge=0.0,
        le=1.0
    )
    max_equations: Optional[int] = Field(
        None,
        description="Maximum number of equations to extract",
        ge=1,
        le=10000
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include equation metadata"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True
    )

    @field_validator('document_content')
    @classmethod
    def validate_document_content(cls, v):
        """Validate document content is not empty."""
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v


class ExtractedEquationResponse(BaseModel):
    """Response model for a single extracted equation."""
    equation_id: str = Field(..., description="Unique equation identifier")
    latex_form: str = Field(..., description="LaTeX form of the equation")
    equation_type: str = Field(..., description="Type of equation (SDE, ODE, etc.)")
    variables: List[str] = Field(default=[], description="Variables in the equation")
    parameters: List[str] = Field(default=[], description="Parameters in the equation")
    confidence_score: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    complexity_score: float = Field(..., description="Complexity score", ge=0.0, le=1.0)
    section_reference: Optional[str] = Field(None, description="Section where equation appears")
    context: str = Field(..., description="Context of the equation")

    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=False  # Preserve exact LaTeX formatting
    )


class ExtractEquationsResponse(APIResponse):
    """Response model for equation extraction endpoint."""
    document_id: str = Field(..., description="Document identifier")
    total_equations: int = Field(..., description="Total number of equations extracted", ge=0)
    equations: List[ExtractedEquationResponse] = Field(
        default=[],
        description="List of extracted equations"
    )
    extraction_statistics: Dict[str, Any] = Field(
        default={},
        description="Extraction statistics and metrics"
    )
    validation_report: Dict[str, Any] = Field(
        default={},
        description="Validation report for mathematical fidelity"
    )


# ===== VARIABLE EXTRACTION MODELS =====

class ExtractVariablesRequest(BaseModel):
    """Request model for variable extraction endpoint."""
    document_content: str = Field(
        ...,
        description="LaTeX document content to process",
        min_length=1,
        max_length=1_000_000
    )
    document_id: Optional[str] = Field(
        None,
        description="Optional document identifier"
    )
    include_dependencies: bool = Field(
        default=True,
        description="Whether to analyze variable dependencies"
    )
    variable_types: Optional[List[str]] = Field(
        None,
        description="Filter by variable types (STATE, PARAMETER, etc.)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Desired output format"
    )

    model_config = ConfigDict(validate_assignment=True)


class ExtractedVariableResponse(BaseModel):
    """Response model for a single extracted variable."""
    symbol: str = Field(..., description="Mathematical symbol")
    variable_type: str = Field(..., description="Type of variable")
    latex_form: str = Field(..., description="LaTeX representation")
    domain: Optional[str] = Field(None, description="Variable domain")
    units: Optional[str] = Field(None, description="Units of measurement")
    dependencies: List[str] = Field(default=[], description="Variable dependencies")
    equation_references: List[str] = Field(default=[], description="Equations using this variable")

    model_config = ConfigDict(str_strip_whitespace=False)


class DependencyGraphResponse(BaseModel):
    """Response model for dependency graph."""
    nodes: List[str] = Field(..., description="Graph nodes (variables)")
    edges: List[List[str]] = Field(..., description="Graph edges (dependencies)")
    evaluation_order: List[str] = Field(..., description="Topological evaluation order")
    cycles: Optional[List[List[str]]] = Field(None, description="Detected cycles")
    strongly_connected_components: List[List[str]] = Field(
        default=[],
        description="Strongly connected components"
    )


class ExtractVariablesResponse(APIResponse):
    """Response model for variable extraction endpoint."""
    document_id: str = Field(..., description="Document identifier")
    total_variables: int = Field(..., description="Total number of variables", ge=0)
    variables: List[ExtractedVariableResponse] = Field(
        default=[],
        description="List of extracted variables"
    )
    dependency_graph: Optional[DependencyGraphResponse] = Field(
        None,
        description="Variable dependency graph"
    )
    analysis_report: Dict[str, Any] = Field(
        default={},
        description="Dependency analysis report"
    )


# ===== SDE MODEL CREATION MODELS =====

class CreateSDEModelRequest(BaseModel):
    """Request model for SDE model creation endpoint."""
    equations: List[str] = Field(
        ...,
        description="List of LaTeX equations defining the SDE",
        min_length=1
    )
    model_name: str = Field(
        ...,
        description="Name for the SDE model",
        min_length=1,
        max_length=100
    )
    variables: Optional[List[str]] = Field(
        None,
        description="State variables (auto-detected if not provided)"
    )
    parameters: Optional[List[str]] = Field(
        None,
        description="Model parameters (auto-detected if not provided)"
    )
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.STANDARD,
        description="Level of mathematical validation"
    )
    auto_classify: bool = Field(
        default=True,
        description="Whether to automatically classify SDE type"
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator('equations')
    @classmethod
    def validate_equations(cls, v):
        """Validate equations list is not empty."""
        if not v:
            raise ValueError("At least one equation must be provided")
        for eq in v:
            if not eq or not eq.strip():
                raise ValueError("Equations cannot be empty")
        return v


class SDEModelResponse(BaseModel):
    """Response model for SDE model information."""
    model_name: str = Field(..., description="SDE model name")
    model_type: str = Field(..., description="Type of SDE model")
    sde_type: str = Field(..., description="Specific SDE classification")
    dimension: int = Field(..., description="Model dimension", ge=1)
    state_variables: List[str] = Field(..., description="State variables")
    drift_functions: List[str] = Field(..., description="Drift function expressions")
    diffusion_functions: List[str] = Field(..., description="Diffusion function expressions")
    jump_components: Optional[List[str]] = Field(None, description="Jump components")
    parameters: List[str] = Field(default=[], description="Model parameters")
    suggested_numerical_scheme: str = Field(..., description="Recommended numerical scheme")
    confidence_score: float = Field(..., description="Classification confidence", ge=0.0, le=1.0)

    model_config = ConfigDict(str_strip_whitespace=False)


class CreateSDEModelResponse(APIResponse):
    """Response model for SDE model creation endpoint."""
    sde_model: SDEModelResponse = Field(..., description="Created SDE model")
    classification_details: Dict[str, Any] = Field(
        default={},
        description="Detailed classification information"
    )
    validation_report: Dict[str, Any] = Field(
        default={},
        description="Model validation report"
    )


# ===== SIMULATION MODELS =====

class SimulationRequest(BaseModel):
    """Request model for SDE simulation endpoint."""
    model_reference: str = Field(
        ...,
        description="Reference to the SDE model to simulate",
        min_length=1
    )
    time_horizon: float = Field(
        ...,
        description="Total simulation time",
        gt=0,
        le=1000
    )
    time_steps: int = Field(
        ...,
        description="Number of time steps",
        gt=0,
        le=1_000_000
    )
    monte_carlo_paths: int = Field(
        ...,
        description="Number of Monte Carlo paths",
        gt=0,
        le=100_000
    )
    numerical_scheme: str = Field(
        default="EULER_MARUYAMA",
        description="Numerical scheme for integration"
    )
    random_seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )
    output_specifications: Optional[List[str]] = Field(
        None,
        description="Specific output requirements"
    )
    performance_targets: Optional[Dict[str, Any]] = Field(
        None,
        description="Performance targets and constraints"
    )

    model_config = ConfigDict(validate_assignment=True)


class SimulationStatistics(BaseModel):
    """Statistical results from simulation."""
    mean: List[float] = Field(..., description="Sample means")
    std: List[float] = Field(..., description="Sample standard deviations")
    min_values: List[float] = Field(..., description="Minimum values")
    max_values: List[float] = Field(..., description="Maximum values")
    percentiles: Dict[str, List[float]] = Field(
        default={},
        description="Percentile values (5%, 25%, 50%, 75%, 95%)"
    )


class SimulationResponse(APIResponse):
    """Response model for simulation endpoint."""
    model_reference: str = Field(..., description="Simulated model reference")
    simulation_results: Dict[str, Any] = Field(
        default={},
        description="Simulation output data"
    )
    statistics: SimulationStatistics = Field(..., description="Statistical analysis")
    performance_metrics: Dict[str, float] = Field(
        default={},
        description="Performance metrics"
    )
    diagnostics: Dict[str, Any] = Field(
        default={},
        description="Simulation diagnostics"
    )


# ===== VALIDATION MODELS =====

class ValidationRequest(BaseModel):
    """Request model for mathematical validation endpoint."""
    model_id: str = Field(
        ...,
        description="Model identifier to validate",
        min_length=1
    )
    validation_tests: List[str] = Field(
        ...,
        description="Types of validation tests to perform",
        min_length=1
    )
    tolerance: float = Field(
        default=1e-6,
        description="Numerical tolerance for validation",
        gt=0,
        le=1.0
    )
    reference_solutions: Optional[Dict[str, Any]] = Field(
        None,
        description="Reference solutions for comparison"
    )
    test_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional test parameters"
    )

    model_config = ConfigDict(validate_assignment=True)


class ValidationTestResult(BaseModel):
    """Result of a single validation test."""
    test_id: str = Field(..., description="Test identifier")
    test_type: str = Field(..., description="Type of validation test")
    passed: bool = Field(..., description="Whether test passed")
    score: float = Field(..., description="Test score", ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default={}, description="Detailed test results")
    error_message: Optional[str] = Field(None, description="Error message if test failed")


class ValidationResponse(APIResponse):
    """Response model for validation endpoint."""
    model_id: str = Field(..., description="Validated model identifier")
    overall_status: str = Field(..., description="Overall validation status")
    overall_score: float = Field(..., description="Overall validation score", ge=0.0, le=1.0)
    test_results: List[ValidationTestResult] = Field(
        default=[],
        description="Individual test results"
    )
    validation_summary: Dict[str, Any] = Field(
        default={},
        description="Validation summary and recommendations"
    )
    constitutional_compliance: bool = Field(
        ...,
        description="Whether mathematical fidelity requirements are met"
    )


# ===== ERROR MODELS =====

class ValidationError(BaseModel):
    """Model for validation errors."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="The invalid value")


class ErrorResponse(APIResponse):
    """Enhanced error response model."""
    error_code: str = Field(..., description="Specific error code")
    error_type: str = Field(..., description="Category of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    validation_errors: Optional[List[ValidationError]] = Field(
        None,
        description="Field-specific validation errors"
    )
    suggested_fix: Optional[str] = Field(None, description="Suggested fix for the error")


# ===== HEALTH CHECK MODELS =====

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)
    uptime: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, str] = Field(
        default={},
        description="Status of individual components"
    )
    memory_usage: Optional[Dict[str, float]] = Field(
        None,
        description="Memory usage statistics"
    )
    performance_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Performance metrics"
    )

    model_config = ConfigDict(validate_assignment=True)


# ===== BATCH PROCESSING MODELS =====

class BatchProcessingRequest(BaseModel):
    """Request model for batch processing."""
    documents: List[Dict[str, str]] = Field(
        ...,
        description="List of documents with content and metadata",
        min_length=1,
        max_length=100
    )
    processing_mode: ProcessingMode = Field(
        default=ProcessingMode.STANDARD,
        description="Processing mode for batch"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Whether to process documents in parallel"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Output format for results"
    )

    model_config = ConfigDict(validate_assignment=True)


class BatchProcessingResponse(APIResponse):
    """Response model for batch processing."""
    total_documents: int = Field(..., description="Total documents processed", ge=0)
    successful_documents: int = Field(..., description="Successfully processed documents", ge=0)
    failed_documents: int = Field(..., description="Failed document processing", ge=0)
    results: List[Dict[str, Any]] = Field(
        default=[],
        description="Processing results for each document"
    )
    batch_statistics: Dict[str, Any] = Field(
        default={},
        description="Overall batch statistics"
    )


# ===== UTILITY FUNCTIONS =====

def create_error_response(
    message: str,
    error_code: str = "UNKNOWN_ERROR",
    error_type: str = "INTERNAL_ERROR",
    processing_time: float = 0.0,
    details: Optional[Dict[str, Any]] = None,
    validation_errors: Optional[List[ValidationError]] = None
) -> ErrorResponse:
    """Create standardized error response."""
    return ErrorResponse(
        success=False,
        message=message,
        processing_time=processing_time,
        error_code=error_code,
        error_type=error_type,
        details=details,
        validation_errors=validation_errors
    )


def validate_mathematical_fidelity(content: str) -> bool:
    """Validate mathematical fidelity of content.

    This is a constitutional requirement check.
    """
    # Basic checks for mathematical content preservation
    if not content or not content.strip():
        return False

    # Check for common LaTeX mathematical constructs
    math_indicators = ['\\', '$', 'equation', 'align', 'frac', 'sum', 'int']
    has_math = any(indicator in content for indicator in math_indicators)

    return has_math