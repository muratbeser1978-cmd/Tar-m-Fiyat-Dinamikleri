"""Equation Extractor with pattern recognition for mathematical content.

This module implements the EquationExtractor class with constitutional
mathematical fidelity requirements for equation extraction and classification.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Set
import re
from dataclasses import dataclass
from enum import Enum
import logging

from ..parsers.latex_parser import LaTeXParser, ParsedExpression, MathematicalStructure

# Configure logging
logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Exception raised when equation extraction fails."""
    pass


class EquationType(str, Enum):
    """Extended enumeration of equation types for extraction."""
    SDE = "SDE"
    ODE = "ODE"
    PDE = "PDE"
    ALGEBRAIC = "ALGEBRAIC"
    CONSTRAINT = "CONSTRAINT"
    INTEGRAL_EQUATION = "INTEGRAL_EQUATION"
    DIFFERENCE_EQUATION = "DIFFERENCE_EQUATION"
    OPTIMIZATION = "OPTIMIZATION"
    BOUNDARY_CONDITION = "BOUNDARY_CONDITION"
    INITIAL_CONDITION = "INITIAL_CONDITION"


class EquationContext(str, Enum):
    """Context where equation appears in document."""
    MAIN_TEXT = "MAIN_TEXT"
    THEOREM = "THEOREM"
    LEMMA = "LEMMA"
    PROPOSITION = "PROPOSITION"
    DEFINITION = "DEFINITION"
    EXAMPLE = "EXAMPLE"
    PROOF = "PROOF"
    APPENDIX = "APPENDIX"
    FOOTNOTE = "FOOTNOTE"


@dataclass
class ExtractedEquation:
    """Represents an extracted equation with comprehensive metadata."""
    equation_id: str
    latex_form: str
    equation_type: EquationType
    context: EquationContext
    section_reference: str
    page_number: Optional[int]
    line_number: Optional[int]
    variables: List[str]
    parameters: List[str]
    functions: List[str]
    is_numbered: bool
    equation_number: Optional[str]
    surrounding_text: str
    dependencies: Dict[str, List[str]]
    complexity_score: float
    confidence_score: float


@dataclass
class ExtractionResult:
    """Results of equation extraction process."""
    document_id: str
    total_equations: int
    equations_by_type: Dict[EquationType, int]
    extracted_equations: List[ExtractedEquation]
    processing_time: float
    success_rate: float
    validation_errors: List[str]


class EquationExtractor:
    """Mathematical equation extractor with pattern recognition.

    This class enforces constitutional mathematical fidelity requirements:
    - Exact preservation of LaTeX equation forms
    - Comprehensive equation classification and metadata extraction
    - Mathematical structure validation and context analysis
    """

    def __init__(self, latex_parser: Optional[LaTeXParser] = None):
        """Initialize equation extractor.

        Args:
            latex_parser: LaTeX parser instance (creates new if None)
        """
        self.latex_parser = latex_parser or LaTeXParser(strict_mode=True)
        self._initialize_patterns()
        self._initialize_classifiers()

    def _initialize_patterns(self):
        """Initialize regex patterns for equation extraction."""
        # Equation environment patterns
        self.equation_environments = {
            'equation': r'\\begin\{equation\}(.*?)\\end\{equation\}',
            'align': r'\\begin\{align\}(.*?)\\end\{align\}',
            'alignat': r'\\begin\{alignat\}\{[^}]+\}(.*?)\\end\{alignat\}',
            'eqnarray': r'\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}',
            'gather': r'\\begin\{gather\}(.*?)\\end\{gather\}',
            'multline': r'\\begin\{multline\}(.*?)\\end\{multline\}',
            'split': r'\\begin\{split\}(.*?)\\end\{split\}',
            'cases': r'\\begin\{cases\}(.*?)\\end\{cases\}'
        }

        # Inline math patterns
        self.inline_patterns = [
            r'\$([^$]+)\$',
            r'\\[(.*?)\\]',
            r'\\((.*?)\\)'
        ]

        # Context patterns
        self.context_patterns = {
            'theorem': r'\\begin\{theorem\}(.*?)\\end\{theorem\}',
            'lemma': r'\\begin\{lemma\}(.*?)\\end\{lemma\}',
            'proposition': r'\\begin\{proposition\}(.*?)\\end\{proposition\}',
            'definition': r'\\begin\{definition\}(.*?)\\end\{definition\}',
            'example': r'\\begin\{example\}(.*?)\\end\{example\}',
            'proof': r'\\begin\{proof\}(.*?)\\end\{proof\}'
        }

        # Section reference patterns
        self.section_pattern = r'\\section\{([^}]+)\}'
        self.subsection_pattern = r'\\subsection\{([^}]+)\}'
        self.label_pattern = r'\\label\{([^}]+)\}'
        self.equation_number_pattern = r'\\tag\{([^}]+)\}'

    def _initialize_classifiers(self):
        """Initialize equation type classifiers."""
        # SDE indicators
        self.sde_indicators = {
            'differential': [r'd[A-Z]', r'dt', r'dW', r'dB', r'dN'],
            'stochastic': [r'dW', r'dB', r'Brownian', r'Wiener', r'stochastic'],
            'noise': [r'\\\\sigma', r'noise', r'random'],
            'drift': [r'\\\\mu', r'drift', r'deterministic']
        }

        # ODE indicators
        self.ode_indicators = {
            'derivative': [r'\\\\frac\{d\}\{dt\}', r'\\\\dot\{', r'\\\\ddot\{', r"x'", r"x''"],
            'differential': [r'\\\\frac\{d[a-zA-Z]\}\{dt\}', r'dx/dt']
        }

        # PDE indicators
        self.pde_indicators = {
            'partial': [r'\\\\partial', r'\\\\frac\{\\\\partial\}\{\\\\partial\}'],
            'mixed_partial': [r'\\\\frac\{\\\\partial\^2\}\{\\\\partial x \\\\partial y\}']
        }

        # Optimization indicators
        self.optimization_indicators = {
            'objective': [r'\\\\max', r'\\\\min', r'\\\\sup', r'\\\\inf', r'argmax', r'argmin'],
            'constraint': [r'\\\\text\{s\.t\.\}', r'subject to', r'such that'],
            'lagrangian': [r'\\\\mathcal\{L\}', r'Lagrangian']
        }

    def extract_equations(self, document: str, document_id: str = "unknown") -> ExtractionResult:
        """Extract all equations from a LaTeX document.

        Args:
            document: LaTeX document content
            document_id: Identifier for the document

        Returns:
            ExtractionResult: Comprehensive extraction results

        Raises:
            ExtractionError: If extraction fails critically
        """
        import time
        start_time = time.time()

        try:
            extracted_equations = []
            validation_errors = []

            # Extract numbered equations from environments
            env_equations = self._extract_environment_equations(document)
            extracted_equations.extend(env_equations)

            # Extract inline equations
            inline_equations = self._extract_inline_equations(document)
            extracted_equations.extend(inline_equations)

            # Classify equation types
            for equation in extracted_equations:
                equation.equation_type = self._classify_equation(equation)
                equation.complexity_score = self._calculate_complexity(equation)
                equation.confidence_score = self._calculate_confidence(equation)

            # Validate extracted equations
            for equation in extracted_equations:
                if not self._validate_equation(equation):
                    validation_errors.append(f"Validation failed for equation {equation.equation_id}")

            # Calculate statistics
            total_equations = len(extracted_equations)
            equations_by_type = self._count_by_type(extracted_equations)
            processing_time = time.time() - start_time
            success_rate = (total_equations - len(validation_errors)) / max(total_equations, 1)

            return ExtractionResult(
                document_id=document_id,
                total_equations=total_equations,
                equations_by_type=equations_by_type,
                extracted_equations=extracted_equations,
                processing_time=processing_time,
                success_rate=success_rate,
                validation_errors=validation_errors
            )

        except Exception as e:
            logger.error(f"Equation extraction failed for document {document_id}: {str(e)}")
            raise ExtractionError(f"Failed to extract equations: {str(e)}")

    def _extract_environment_equations(self, document: str) -> List[ExtractedEquation]:
        """Extract equations from LaTeX environments."""
        equations = []
        equation_counter = 1

        for env_name, pattern in self.equation_environments.items():
            matches = re.finditer(pattern, document, re.DOTALL)

            for match in matches:
                equation_content = match.group(1).strip()
                context = self._determine_context(document, match.start(), match.end())
                section_ref = self._find_section_reference(document, match.start())

                # Check for equation numbering
                is_numbered = env_name in ['equation', 'align', 'eqnarray', 'gather']
                equation_number = self._extract_equation_number(equation_content)

                # Parse the equation
                try:
                    parsed = self.latex_parser.parse_latex(equation_content)

                    equation = ExtractedEquation(
                        equation_id=f"eq_{equation_counter}",
                        latex_form=equation_content,
                        equation_type=EquationType.ALGEBRAIC,  # Will be classified later
                        context=context,
                        section_reference=section_ref,
                        page_number=None,  # Would need additional processing
                        line_number=document[:match.start()].count('\n') + 1,
                        variables=parsed.variables,
                        parameters=parsed.parameters,
                        functions=parsed.functions,
                        is_numbered=is_numbered,
                        equation_number=equation_number,
                        surrounding_text=self._extract_surrounding_text(document, match.start(), match.end()),
                        dependencies=parsed.dependencies,
                        complexity_score=0.0,  # Will be calculated later
                        confidence_score=0.0   # Will be calculated later
                    )

                    equations.append(equation)
                    equation_counter += 1

                except Exception as e:
                    logger.warning(f"Failed to parse equation: {equation_content}, Error: {str(e)}")

        return equations

    def _extract_inline_equations(self, document: str) -> List[ExtractedEquation]:
        """Extract inline mathematical expressions."""
        equations = []
        equation_counter = 1000  # Use different numbering for inline

        for pattern in self.inline_patterns:
            matches = re.finditer(pattern, document, re.DOTALL)

            for match in matches:
                equation_content = match.group(1).strip()

                # Skip very short expressions that are likely just variables
                if len(equation_content) < 3:
                    continue

                # Skip simple variable references
                if re.match(r'^[a-zA-Z]$', equation_content):
                    continue

                context = self._determine_context(document, match.start(), match.end())
                section_ref = self._find_section_reference(document, match.start())

                try:
                    parsed = self.latex_parser.parse_latex(equation_content)

                    equation = ExtractedEquation(
                        equation_id=f"inline_{equation_counter}",
                        latex_form=equation_content,
                        equation_type=EquationType.ALGEBRAIC,
                        context=context,
                        section_reference=section_ref,
                        page_number=None,
                        line_number=document[:match.start()].count('\n') + 1,
                        variables=parsed.variables,
                        parameters=parsed.parameters,
                        functions=parsed.functions,
                        is_numbered=False,
                        equation_number=None,
                        surrounding_text=self._extract_surrounding_text(document, match.start(), match.end()),
                        dependencies=parsed.dependencies,
                        complexity_score=0.0,
                        confidence_score=0.0
                    )

                    equations.append(equation)
                    equation_counter += 1

                except Exception as e:
                    logger.warning(f"Failed to parse inline equation: {equation_content}, Error: {str(e)}")

        return equations

    def _classify_equation(self, equation: ExtractedEquation) -> EquationType:
        """Classify equation type based on mathematical content."""
        latex = equation.latex_form.lower()

        # Check for SDE indicators
        if self._contains_indicators(latex, self.sde_indicators):
            return EquationType.SDE

        # Check for ODE indicators
        if self._contains_indicators(latex, self.ode_indicators):
            return EquationType.ODE

        # Check for PDE indicators
        if self._contains_indicators(latex, self.pde_indicators):
            return EquationType.PDE

        # Check for optimization indicators
        if self._contains_indicators(latex, self.optimization_indicators):
            return EquationType.OPTIMIZATION

        # Check for integral equations
        if 'int' in latex and any(var in latex for var in equation.variables):
            return EquationType.INTEGRAL_EQUATION

        # Check for boundary/initial conditions
        if any(term in latex for term in ['boundary', 'initial', 't=0', 'x=0']):
            if 'boundary' in equation.surrounding_text.lower():
                return EquationType.BOUNDARY_CONDITION
            else:
                return EquationType.INITIAL_CONDITION

        # Check for constraints
        if any(term in latex for term in ['leq', 'geq', 'subject', 'constraint']):
            return EquationType.CONSTRAINT

        # Default to algebraic
        return EquationType.ALGEBRAIC

    def _contains_indicators(self, latex: str, indicator_dict: Dict[str, List[str]]) -> bool:
        """Check if latex contains any indicators from the given dictionary."""
        for category, indicators in indicator_dict.items():
            for indicator in indicators:
                if re.search(indicator, latex):
                    return True
        return False

    def _determine_context(self, document: str, start_pos: int, end_pos: int) -> EquationContext:
        """Determine the context where the equation appears."""
        # Look backwards and forwards for context markers
        context_window = 2000  # characters
        window_start = max(0, start_pos - context_window)
        window_end = min(len(document), end_pos + context_window)
        context_text = document[window_start:window_end].lower()

        # Check for theorem-like environments
        for context_name, pattern in self.context_patterns.items():
            if re.search(pattern, context_text, re.DOTALL):
                return EquationContext[context_name.upper()]

        # Check for appendix
        if 'appendix' in context_text:
            return EquationContext.APPENDIX

        # Default to main text
        return EquationContext.MAIN_TEXT

    def _find_section_reference(self, document: str, position: int) -> str:
        """Find the section reference for the equation."""
        # Look backwards for the most recent section or subsection
        text_before = document[:position]

        # Find the last section
        section_matches = list(re.finditer(self.section_pattern, text_before))
        subsection_matches = list(re.finditer(self.subsection_pattern, text_before))

        all_matches = [(m, 'section') for m in section_matches] + [(m, 'subsection') for m in subsection_matches]
        all_matches.sort(key=lambda x: x[0].start())

        if all_matches:
            last_match, match_type = all_matches[-1]
            return f"{match_type}: {last_match.group(1)}"

        return "Unknown section"

    def _extract_equation_number(self, equation_content: str) -> Optional[str]:
        """Extract equation number if present."""
        match = re.search(self.equation_number_pattern, equation_content)
        return match.group(1) if match else None

    def _extract_surrounding_text(self, document: str, start_pos: int, end_pos: int) -> str:
        """Extract surrounding text for context."""
        window_size = 200
        window_start = max(0, start_pos - window_size)
        window_end = min(len(document), end_pos + window_size)

        before_text = document[window_start:start_pos].strip()
        after_text = document[end_pos:window_end].strip()

        return f"{before_text} ... {after_text}"

    def _calculate_complexity(self, equation: ExtractedEquation) -> float:
        """Calculate complexity score for the equation."""
        complexity = 0.0

        # Base complexity from length
        complexity += len(equation.latex_form) * 0.01

        # Add complexity for number of variables and parameters
        complexity += len(equation.variables) * 0.5
        complexity += len(equation.parameters) * 0.3
        complexity += len(equation.functions) * 0.4

        # Add complexity for mathematical operations
        operations = ['frac', 'int', 'sum', 'prod', 'partial', 'sqrt']
        for op in operations:
            complexity += equation.latex_form.count(op) * 0.2

        # Normalize to 0-1 scale
        return min(complexity / 10.0, 1.0)

    def _calculate_confidence(self, equation: ExtractedEquation) -> float:
        """Calculate confidence score for the extraction."""
        confidence = 1.0

        # Reduce confidence for very short equations
        if len(equation.latex_form) < 5:
            confidence *= 0.7

        # Reduce confidence if no variables or parameters found
        if not equation.variables and not equation.parameters:
            confidence *= 0.5

        # Increase confidence for numbered equations
        if equation.is_numbered:
            confidence *= 1.1

        return min(confidence, 1.0)

    def _validate_equation(self, equation: ExtractedEquation) -> bool:
        """Validate extracted equation for constitutional compliance."""
        # CONSTITUTIONAL REQUIREMENT: LaTeX form must be preserved exactly
        if not equation.latex_form:
            return False

        # CONSTITUTIONAL REQUIREMENT: Must have valid equation ID
        if not equation.equation_id:
            return False

        # Basic structure validation
        try:
            parsed = self.latex_parser.parse_latex(equation.latex_form)
            return self.latex_parser.validate_constitutional_compliance(parsed)
        except Exception:
            return False

    def _count_by_type(self, equations: List[ExtractedEquation]) -> Dict[EquationType, int]:
        """Count equations by type."""
        counts = {}
        for equation in equations:
            counts[equation.equation_type] = counts.get(equation.equation_type, 0) + 1
        return counts

    def filter_equations(self,
                        equations: List[ExtractedEquation],
                        equation_types: Optional[List[EquationType]] = None,
                        min_complexity: float = 0.0,
                        min_confidence: float = 0.0) -> List[ExtractedEquation]:
        """Filter equations based on criteria."""
        filtered = equations

        if equation_types:
            filtered = [eq for eq in filtered if eq.equation_type in equation_types]

        if min_complexity > 0:
            filtered = [eq for eq in filtered if eq.complexity_score >= min_complexity]

        if min_confidence > 0:
            filtered = [eq for eq in filtered if eq.confidence_score >= min_confidence]

        return filtered

    def validate_constitutional_compliance(self, extraction_result: ExtractionResult) -> bool:
        """Validate constitutional mathematical fidelity requirements."""
        # CONSTITUTIONAL REQUIREMENT: All equations must preserve exact LaTeX forms
        for equation in extraction_result.extracted_equations:
            if not self._validate_equation(equation):
                return False

        return True