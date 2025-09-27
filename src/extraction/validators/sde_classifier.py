"""SDE Model Classifier and Structure Analyzer.

This module implements the SDEClassifier class with constitutional
mathematical fidelity requirements for SDE model classification
and mathematical structure analysis.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re
import logging
import numpy as np

from ..parsers.latex_parser import LaTeXParser, ParsedExpression
from ..extractors.equation_extractor import ExtractedEquation, EquationType
from ..validators.dependency_analyzer import DependencyAnalyzer, DependencyAnalysisResult
from ...sde.models.sde_model import SDEModel, ModelType

# Configure logging
logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """Exception raised when SDE classification fails."""
    pass


class SDEType(str, Enum):
    """Detailed SDE classification types."""
    GEOMETRIC_BROWNIAN_MOTION = "GEOMETRIC_BROWNIAN_MOTION"
    ORNSTEIN_UHLENBECK = "ORNSTEIN_UHLENBECK"
    COX_INGERSOLL_ROSS = "COX_INGERSOLL_ROSS"
    VASICEK = "VASICEK"
    HESTON = "HESTON"
    JUMP_DIFFUSION = "JUMP_DIFFUSION"
    FRACTIONAL_BROWNIAN_MOTION = "FRACTIONAL_BROWNIAN_MOTION"
    LEVY_PROCESS = "LEVY_PROCESS"
    MULTIDIMENSIONAL = "MULTIDIMENSIONAL"
    CUSTOM = "CUSTOM"


class DriftType(str, Enum):
    """Types of drift functions."""
    LINEAR = "LINEAR"           # μ(X,t) = aX + b
    NONLINEAR = "NONLINEAR"     # μ(X,t) = f(X,t)
    MEAN_REVERTING = "MEAN_REVERTING"  # μ(X,t) = κ(θ - X)
    EXPONENTIAL = "EXPONENTIAL"  # μ(X,t) = μX
    POLYNOMIAL = "POLYNOMIAL"    # μ(X,t) = Σ aᵢXⁱ
    CUSTOM = "CUSTOM"


class DiffusionType(str, Enum):
    """Types of diffusion functions."""
    CONSTANT = "CONSTANT"       # σ(X,t) = σ
    LINEAR = "LINEAR"           # σ(X,t) = σX
    SQUARE_ROOT = "SQUARE_ROOT" # σ(X,t) = σ√X
    NONLINEAR = "NONLINEAR"     # σ(X,t) = f(X,t)
    STOCHASTIC_VOLATILITY = "STOCHASTIC_VOLATILITY"
    CUSTOM = "CUSTOM"


@dataclass
class SDEStructure:
    """Detailed structure analysis of an SDE."""
    sde_type: SDEType
    model_type: ModelType
    drift_type: DriftType
    diffusion_type: DiffusionType
    state_variables: List[str]
    noise_variables: List[str]
    parameters: List[str]
    dimension: int
    has_jumps: bool
    jump_intensity: Optional[str]
    is_time_homogeneous: bool
    is_autonomous: bool
    boundary_behavior: Optional[str]
    analytical_solution_available: bool


@dataclass
class ClassificationResult:
    """Result of SDE classification and analysis."""
    equation_id: str
    original_latex: str
    sde_structure: SDEStructure
    confidence_score: float
    drift_function: str
    diffusion_function: str
    jump_components: List[str]
    model_parameters: Dict[str, Any]
    validation_errors: List[str]
    suggested_numerical_scheme: str


class SDEClassifier:
    """SDE model classifier and structure analyzer.

    This class enforces constitutional mathematical fidelity requirements:
    - Exact preservation of SDE mathematical expressions
    - Comprehensive classification of SDE types and structures
    - Mathematical validation of SDE properties and constraints
    """

    def __init__(self, latex_parser: Optional[LaTeXParser] = None,
                 dependency_analyzer: Optional[DependencyAnalyzer] = None):
        """Initialize SDE classifier.

        Args:
            latex_parser: LaTeX parser instance
            dependency_analyzer: Dependency analyzer instance
        """
        self.latex_parser = latex_parser or LaTeXParser(strict_mode=True)
        self.dependency_analyzer = dependency_analyzer or DependencyAnalyzer(self.latex_parser)
        self._initialize_patterns()
        self._initialize_classifiers()

    def _initialize_patterns(self):
        """Initialize regex patterns for SDE classification."""
        # Standard SDE patterns
        self.sde_patterns = {
            'geometric_brownian_motion': [
                r'dS\s*=\s*\\mu\s*S\s*dt\s*\+\s*\\sigma\s*S\s*dW',
                r'dX\s*=\s*\\mu\s*X\s*dt\s*\+\s*\\sigma\s*X\s*dW'
            ],
            'ornstein_uhlenbeck': [
                r'd[XY]\s*=\s*\\theta\s*\(\\mu\s*-\s*[XY]\)\s*dt\s*\+\s*\\sigma\s*dW',
                r'd[XY]\s*=\s*-\\theta\s*[XY]\s*dt\s*\+\s*\\sigma\s*dW'
            ],
            'cox_ingersoll_ross': [
                r'dr\s*=\s*\\kappa\s*\(\\theta\s*-\s*r\)\s*dt\s*\+\s*\\sigma\s*\\sqrt\{r\}\s*dW'
            ],
            'vasicek': [
                r'dr\s*=\s*a\s*\(b\s*-\s*r\)\s*dt\s*\+\s*\\sigma\s*dW'
            ],
            'heston': [
                r'dS\s*=.*dW_1.*dv\s*=.*dW_2',
                r'\\text{Heston}|\\text{stochastic volatility}'
            ]
        }

        # Drift function patterns
        self.drift_patterns = {
            'linear': r'([+-]?\s*\\?[a-zA-Z_]+\s*[XYSr]?\s*[+-]?\s*\\?[a-zA-Z_]*)',
            'mean_reverting': r'\\?[a-zA-Z_]+\s*\(\s*\\?[a-zA-Z_]+\s*-\s*[XYSr]\s*\)',
            'exponential': r'\\?[a-zA-Z_]+\s*[XYSr]',
            'polynomial': r'[XYSr]\^?\{?[2-9]\}?'
        }

        # Diffusion function patterns
        self.diffusion_patterns = {
            'constant': r'^\\?[a-zA-Z_]+$',
            'linear': r'\\?[a-zA-Z_]+\s*[XYSr]',
            'square_root': r'\\?[a-zA-Z_]+\s*\\sqrt\{[XYSr]\}',
            'nonlinear': r'\\?[a-zA-Z_]+\s*[XYSr]\^?\{?[2-9]\}?'
        }

        # Jump process indicators
        self.jump_indicators = [
            'dN', 'dJ', 'Poisson', 'jump', 'compound Poisson',
            'Lévy', 'Levy', 'α-stable', 'alpha-stable'
        ]

        # Time dependency patterns
        self.time_patterns = [
            r'[XYSr]_\{t\}', r'[XYSr]\(t\)', r'\\mu\(t\)', r'\\sigma\(t\)'
        ]

    def _initialize_classifiers(self):
        """Initialize classification parameters."""
        # Parameter patterns for known models
        self.parameter_mappings = {
            SDEType.GEOMETRIC_BROWNIAN_MOTION: ['\\mu', '\\sigma'],
            SDEType.ORNSTEIN_UHLENBECK: ['\\theta', '\\mu', '\\sigma'],
            SDEType.COX_INGERSOLL_ROSS: ['\\kappa', '\\theta', '\\sigma'],
            SDEType.VASICEK: ['a', 'b', '\\sigma'],
            SDEType.HESTON: ['\\mu', '\\kappa', '\\theta', '\\sigma', '\\rho']
        }

        # Numerical scheme recommendations
        self.scheme_recommendations = {
            SDEType.GEOMETRIC_BROWNIAN_MOTION: "MILSTEIN",
            SDEType.ORNSTEIN_UHLENBECK: "EULER_MARUYAMA",
            SDEType.COX_INGERSOLL_ROSS: "MILSTEIN",
            SDEType.JUMP_DIFFUSION: "EULER_MARUYAMA",
            SDEType.CUSTOM: "EULER_MARUYAMA"
        }

    def classify_sde(self, equation: ExtractedEquation) -> ClassificationResult:
        """Classify an SDE equation and analyze its structure.

        Args:
            equation: Extracted equation to classify

        Returns:
            ClassificationResult: Comprehensive classification results

        Raises:
            ClassificationError: If classification fails
        """
        try:
            # Parse the equation
            parsed = self.latex_parser.parse_latex(equation.latex_form)

            # Determine if it's actually an SDE
            if not self._is_sde(equation, parsed):
                raise ClassificationError(f"Equation {equation.equation_id} is not an SDE")

            # Classify SDE type
            sde_type = self._classify_sde_type(equation, parsed)

            # Analyze SDE structure
            sde_structure = self._analyze_sde_structure(equation, parsed, sde_type)

            # Extract components
            drift_function, diffusion_function, jump_components = self._extract_sde_components(equation, parsed)

            # Calculate confidence score
            confidence_score = self._calculate_classification_confidence(sde_type, sde_structure, equation)

            # Extract model parameters
            model_parameters = self._extract_model_parameters(equation, parsed, sde_type)

            # Validate classification
            validation_errors = self._validate_classification(equation, sde_structure)

            # Suggest numerical scheme
            suggested_scheme = self.scheme_recommendations.get(sde_type, "EULER_MARUYAMA")

            return ClassificationResult(
                equation_id=equation.equation_id,
                original_latex=equation.latex_form,
                sde_structure=sde_structure,
                confidence_score=confidence_score,
                drift_function=drift_function,
                diffusion_function=diffusion_function,
                jump_components=jump_components,
                model_parameters=model_parameters,
                validation_errors=validation_errors,
                suggested_numerical_scheme=suggested_scheme
            )

        except Exception as e:
            logger.error(f"SDE classification failed for equation {equation.equation_id}: {str(e)}")
            raise ClassificationError(f"Failed to classify SDE: {str(e)}")

    def _is_sde(self, equation: ExtractedEquation, parsed: ParsedExpression) -> bool:
        """Determine if equation is a stochastic differential equation."""
        # Must be classified as SDE type
        if equation.equation_type != EquationType.SDE:
            return False

        # Must contain stochastic differential terms
        if not parsed.is_stochastic:
            return False

        # Must contain differential terms
        if not parsed.differential_terms:
            return False

        # Must contain stochastic noise terms
        latex_lower = equation.latex_form.lower()
        stochastic_terms = ['dw', 'db', 'dn', 'wiener', 'brownian']
        return any(term in latex_lower for term in stochastic_terms)

    def _classify_sde_type(self, equation: ExtractedEquation, parsed: ParsedExpression) -> SDEType:
        """Classify the specific type of SDE."""
        latex = equation.latex_form

        # Check against known SDE patterns
        for sde_name, patterns in self.sde_patterns.items():
            for pattern in patterns:
                if re.search(pattern, latex, re.IGNORECASE):
                    return SDEType[sde_name.upper()]

        # Check for jump processes
        if any(indicator in latex for indicator in self.jump_indicators):
            return SDEType.JUMP_DIFFUSION

        # Check for multidimensional systems
        if self._is_multidimensional(equation, parsed):
            return SDEType.MULTIDIMENSIONAL

        # Default to custom
        return SDEType.CUSTOM

    def _is_multidimensional(self, equation: ExtractedEquation, parsed: ParsedExpression) -> bool:
        """Check if SDE is multidimensional."""
        # Count unique state variables
        state_vars = set()
        for var in parsed.variables:
            # Look for patterns like X_1, X_2, Y, Z
            if re.match(r'[A-Z](_\{\d+\})?$', var):
                state_vars.add(var)

        return len(state_vars) > 1

    def _analyze_sde_structure(self, equation: ExtractedEquation, parsed: ParsedExpression, sde_type: SDEType) -> SDEStructure:
        """Analyze detailed SDE structure."""
        # Extract drift and diffusion types
        drift_function, diffusion_function, _ = self._extract_sde_components(equation, parsed)
        drift_type = self._classify_drift_type(drift_function)
        diffusion_type = self._classify_diffusion_type(diffusion_function)

        # Determine model type
        if sde_type == SDEType.MULTIDIMENSIONAL:
            model_type = ModelType.MULTIVARIATE
        elif sde_type == SDEType.JUMP_DIFFUSION:
            model_type = ModelType.JUMP_DIFFUSION
        else:
            model_type = ModelType.SCALAR

        # Extract variables
        state_variables = [var for var in parsed.variables if self._is_state_variable(var)]
        noise_variables = [var for var in parsed.variables if self._is_noise_variable(var)]

        # For SCALAR SDE, ensure only one state variable
        if model_type == ModelType.SCALAR and len(state_variables) != 1:
            # Take the first reasonable state variable or create a default
            if state_variables:
                state_variables = [state_variables[0]]
            else:
                # Default state variable for unidentified cases
                state_variables = ["X"]

        # Check properties
        has_jumps = sde_type == SDEType.JUMP_DIFFUSION or any(indicator in equation.latex_form for indicator in self.jump_indicators)
        is_time_homogeneous = not any(re.search(pattern, equation.latex_form) for pattern in self.time_patterns)
        is_autonomous = 't' not in equation.latex_form or not re.search(r'[^d]t[^_]', equation.latex_form)

        # Check for analytical solutions
        analytical_solution_available = sde_type in [
            SDEType.GEOMETRIC_BROWNIAN_MOTION,
            SDEType.ORNSTEIN_UHLENBECK,
            SDEType.VASICEK
        ]

        return SDEStructure(
            sde_type=sde_type,
            model_type=model_type,
            drift_type=drift_type,
            diffusion_type=diffusion_type,
            state_variables=state_variables,
            noise_variables=noise_variables,
            parameters=parsed.parameters,
            dimension=len(state_variables),
            has_jumps=has_jumps,
            jump_intensity=self._extract_jump_intensity(equation) if has_jumps else None,
            is_time_homogeneous=is_time_homogeneous,
            is_autonomous=is_autonomous,
            boundary_behavior=self._analyze_boundary_behavior(equation, sde_type),
            analytical_solution_available=analytical_solution_available
        )

    def _extract_sde_components(self, equation: ExtractedEquation, parsed: ParsedExpression) -> Tuple[str, str, List[str]]:
        """Extract drift, diffusion, and jump components from SDE."""
        latex = equation.latex_form

        # Split SDE into components using 'dt' and 'dW' terms
        drift_function = ""
        diffusion_function = ""
        jump_components = []

        # Find drift term (coefficient of dt)
        dt_pattern = r'([^+\-=]*)\s*dt'
        dt_matches = re.findall(dt_pattern, latex)
        if dt_matches:
            drift_function = dt_matches[0].strip()

        # Find diffusion term (coefficient of dW)
        dw_pattern = r'([^+\-=]*)\s*d[WB]'
        dw_matches = re.findall(dw_pattern, latex)
        if dw_matches:
            diffusion_function = dw_matches[0].strip()

        # Find jump terms (coefficient of dN)
        dn_pattern = r'([^+\-=]*)\s*d[NJ]'
        dn_matches = re.findall(dn_pattern, latex)
        jump_components = [match.strip() for match in dn_matches]

        return drift_function, diffusion_function, jump_components

    def _classify_drift_type(self, drift_function: str) -> DriftType:
        """Classify the type of drift function."""
        if not drift_function:
            return DriftType.CUSTOM

        # Check for mean-reverting pattern
        if re.search(r'\\?[a-zA-Z_]+\s*\([^)]*-[^)]*\)', drift_function):
            return DriftType.MEAN_REVERTING

        # Check for linear pattern
        if re.search(self.drift_patterns['linear'], drift_function):
            return DriftType.LINEAR

        # Check for exponential pattern
        if re.search(self.drift_patterns['exponential'], drift_function):
            return DriftType.EXPONENTIAL

        # Check for polynomial pattern
        if re.search(self.drift_patterns['polynomial'], drift_function):
            return DriftType.POLYNOMIAL

        return DriftType.CUSTOM

    def _classify_diffusion_type(self, diffusion_function: str) -> DiffusionType:
        """Classify the type of diffusion function."""
        if not diffusion_function:
            return DiffusionType.CUSTOM

        # Check for constant
        if re.search(self.diffusion_patterns['constant'], diffusion_function):
            return DiffusionType.CONSTANT

        # Check for square root
        if re.search(self.diffusion_patterns['square_root'], diffusion_function):
            return DiffusionType.SQUARE_ROOT

        # Check for linear
        if re.search(self.diffusion_patterns['linear'], diffusion_function):
            return DiffusionType.LINEAR

        # Check for nonlinear
        if re.search(self.diffusion_patterns['nonlinear'], diffusion_function):
            return DiffusionType.NONLINEAR

        return DiffusionType.CUSTOM

    def _is_state_variable(self, variable: str) -> bool:
        """Check if variable is a state variable."""
        # State variables are usually single capital letters or X_t patterns
        state_patterns = [r'^[XYZS]$', r'^[XYZS]_\{[^}]*\}$', r'^[a-z]$']
        return any(re.match(pattern, variable) for pattern in state_patterns)

    def _is_noise_variable(self, variable: str) -> bool:
        """Check if variable is a noise variable."""
        noise_patterns = [r'^[WB]$', r'^[WB]_\{[^}]*\}$', r'^dW', r'^dB', r'^N']
        return any(re.match(pattern, variable) for pattern in noise_patterns)

    def _extract_jump_intensity(self, equation: ExtractedEquation) -> Optional[str]:
        """Extract jump intensity parameter if present."""
        latex = equation.latex_form

        # Look for common jump intensity patterns
        intensity_patterns = [r'\\lambda', r'\\nu', r'intensity']
        for pattern in intensity_patterns:
            if pattern in latex:
                return pattern

        return None

    def _analyze_boundary_behavior(self, equation: ExtractedEquation, sde_type: SDEType) -> Optional[str]:
        """Analyze boundary behavior of the SDE."""
        # This is a simplified analysis - could be enhanced
        if sde_type == SDEType.COX_INGERSOLL_ROSS:
            return "absorbing_at_zero"
        elif sde_type == SDEType.ORNSTEIN_UHLENBECK:
            return "mean_reverting"
        elif sde_type == SDEType.GEOMETRIC_BROWNIAN_MOTION:
            return "no_boundary"

        return None

    def _extract_model_parameters(self, equation: ExtractedEquation, parsed: ParsedExpression, sde_type: SDEType) -> Dict[str, Any]:
        """Extract model parameters specific to the SDE type."""
        parameters = {}

        # Get expected parameters for this SDE type
        expected_params = self.parameter_mappings.get(sde_type, [])

        for param in parsed.parameters:
            if param in expected_params or any(exp in param for exp in expected_params):
                parameters[param] = {
                    'symbol': param,
                    'type': 'parameter',
                    'domain': 'real',  # Could be enhanced with domain analysis
                    'constraints': self._get_parameter_constraints(param, sde_type)
                }

        return parameters

    def _get_parameter_constraints(self, parameter: str, sde_type: SDEType) -> List[str]:
        """Get constraints for specific parameters in known SDE types."""
        constraints = []

        # Common parameter constraints
        if parameter in ['\\sigma', 'sigma']:
            constraints.append('> 0')  # Volatility must be positive

        if sde_type == SDEType.COX_INGERSOLL_ROSS:
            if parameter in ['\\kappa', 'kappa']:
                constraints.append('> 0')  # Mean reversion speed
            if parameter in ['\\theta', 'theta']:
                constraints.append('> 0')  # Long-term mean

        return constraints

    def _calculate_classification_confidence(self, sde_type: SDEType, structure: SDEStructure, equation: ExtractedEquation) -> float:
        """Calculate confidence in the classification."""
        confidence = 0.5  # Base confidence

        # Increase confidence for known SDE types
        if sde_type != SDEType.CUSTOM:
            confidence += 0.3

        # Increase confidence if structure is well-defined
        if structure.state_variables and structure.noise_variables:
            confidence += 0.1

        # Increase confidence if parameters are identified
        if structure.parameters:
            confidence += 0.1

        # Decrease confidence for very short equations
        if len(equation.latex_form) < 20:
            confidence -= 0.2

        return min(max(confidence, 0.0), 1.0)

    def _validate_classification(self, equation: ExtractedEquation, structure: SDEStructure) -> List[str]:
        """Validate the SDE classification."""
        validation_errors = []

        # Check basic SDE requirements
        if not structure.state_variables:
            validation_errors.append("No state variables identified")

        if not structure.noise_variables:
            validation_errors.append("No noise variables identified")

        # Check dimension consistency
        if structure.model_type == ModelType.SCALAR and structure.dimension != 1:
            validation_errors.append("Scalar SDE should have dimension 1")

        if structure.model_type == ModelType.MULTIVARIATE and structure.dimension <= 1:
            validation_errors.append("Multivariate SDE should have dimension > 1")

        # Check jump components for jump-diffusion
        if structure.model_type == ModelType.JUMP_DIFFUSION and not structure.has_jumps:
            validation_errors.append("Jump-diffusion model should have jump components")

        return validation_errors

    def create_sde_model(self, classification_result: ClassificationResult) -> SDEModel:
        """Create SDEModel instance from classification result.

        Args:
            classification_result: Result of SDE classification

        Returns:
            SDEModel: Validated SDE model instance

        Raises:
            ClassificationError: If model creation fails
        """
        try:
            structure = classification_result.sde_structure

            # Prepare model components
            model_name = f"{structure.sde_type.value}_{classification_result.equation_id}"
            drift_functions = [classification_result.drift_function] if classification_result.drift_function else ["0"]
            diffusion_functions = [classification_result.diffusion_function] if classification_result.diffusion_function else ["1"]
            state_variables = structure.state_variables or ["X"]

            # Create SDEModel
            sde_model = SDEModel(
                model_name=model_name,
                model_type=structure.model_type,
                drift_functions=drift_functions,
                diffusion_functions=diffusion_functions,
                state_variables=state_variables,
                jump_components=classification_result.jump_components if structure.has_jumps else None
            )

            return sde_model

        except Exception as e:
            logger.error(f"Failed to create SDEModel from classification: {str(e)}")
            raise ClassificationError(f"Model creation failed: {str(e)}")

    def validate_constitutional_compliance(self, classification_result: ClassificationResult) -> bool:
        """Validate constitutional mathematical fidelity requirements."""
        # CONSTITUTIONAL REQUIREMENT: Original LaTeX preserved exactly
        if not classification_result.original_latex:
            return False

        # CONSTITUTIONAL REQUIREMENT: Mathematical structure preserved
        if not classification_result.sde_structure.state_variables:
            return False

        # CONSTITUTIONAL REQUIREMENT: No validation errors in strict mode
        if self.latex_parser.strict_mode and classification_result.validation_errors:
            return False

        return True