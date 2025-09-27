"""LaTeX Parser with SymPy integration for mathematical expression parsing.

This module implements the LaTeXParser class with constitutional mathematical
fidelity requirements for exact preservation of mathematical expressions.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import re
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Exception raised when LaTeX parsing fails."""
    pass


class MathematicalStructure(str, Enum):
    """Enumeration of mathematical structure types."""
    EQUATION = "EQUATION"
    INEQUALITY = "INEQUALITY"
    DIFFERENTIAL = "DIFFERENTIAL"
    INTEGRAL = "INTEGRAL"
    SUMMATION = "SUMMATION"
    MATRIX = "MATRIX"
    FUNCTION = "FUNCTION"
    EXPRESSION = "EXPRESSION"


@dataclass
class ParsedSymbol:
    """Represents a parsed mathematical symbol with metadata."""
    symbol: str
    latex_form: str
    symbol_type: str  # variable, parameter, function, operator
    subscripts: List[str]
    superscripts: List[str]
    modifiers: List[str]  # bar, hat, tilde, etc.


@dataclass
class ParsedExpression:
    """Represents a parsed mathematical expression."""
    original_latex: str
    structure_type: MathematicalStructure
    symbols: List[ParsedSymbol]
    operators: List[str]
    functions: List[str]
    parameters: List[str]
    variables: List[str]
    dependencies: Dict[str, List[str]]
    is_stochastic: bool
    differential_terms: List[str]


class LaTeXParser:
    """LaTeX parser with SymPy integration and mathematical fidelity preservation.

    This class enforces constitutional mathematical fidelity requirements:
    - Exact preservation of LaTeX mathematical expressions
    - Comprehensive symbol extraction and classification
    - Mathematical structure recognition and validation
    """

    def __init__(self, strict_mode: bool = True):
        """Initialize LaTeX parser.

        Args:
            strict_mode: If True, enforce strict mathematical fidelity requirements
        """
        self.strict_mode = strict_mode
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize regex patterns for LaTeX parsing."""
        # Core mathematical patterns
        self.symbol_pattern = re.compile(r'\\?([a-zA-Z]+(?:_\{[^}]+\})?(?:\^\{[^}]+\})?)')
        self.subscript_pattern = re.compile(r'_\{([^}]+)\}')
        self.superscript_pattern = re.compile(r'\^\{([^}]+)\}')
        self.fraction_pattern = re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}')
        self.differential_pattern = re.compile(r'd([a-zA-Z_]+(?:_\{[^}]+\})?)')
        self.partial_pattern = re.compile(r'\\partial\s*([a-zA-Z_]+(?:_\{[^}]+\})?)')
        self.integral_pattern = re.compile(r'\\int(?:_\{([^}]+)\})?\^?\{?([^}]*)\}?')
        self.sum_pattern = re.compile(r'\\sum(?:_\{([^}]+)\})?\^?\{?([^}]*)\}?')

        # Greek letters and special symbols
        self.greek_letters = {
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
            'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma',
            'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
            'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
            'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Pi', 'Rho', 'Sigma',
            'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega'
        }

        # Mathematical functions
        self.math_functions = {
            'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs',
            'max', 'min', 'sup', 'inf', 'lim', 'det', 'tr', 'dim'
        }

        # Stochastic terms
        self.stochastic_indicators = {
            'dW', 'dB', 'dN', 'mathbb{E}', 'mathbb{P}', 'Var', 'Cov'
        }

    def parse_latex(self, latex_expression: str) -> ParsedExpression:
        """Parse LaTeX mathematical expression with constitutional fidelity.

        Args:
            latex_expression: LaTeX mathematical expression to parse

        Returns:
            ParsedExpression: Comprehensive parsing results

        Raises:
            ParseError: If parsing fails or violates mathematical fidelity
        """
        if not latex_expression or not isinstance(latex_expression, str):
            raise ParseError("LaTeX expression must be a non-empty string")

        try:
            # Preserve original exactly
            original_latex = latex_expression.strip()

            # Determine mathematical structure type
            structure_type = self._classify_structure(original_latex)

            # Extract symbols with metadata
            symbols = self._extract_symbols(original_latex)

            # Extract operators and functions
            operators = self._extract_operators(original_latex)
            functions = self._extract_functions(original_latex)

            # Classify symbols into parameters and variables
            parameters, variables = self._classify_symbols(symbols)

            # Analyze dependencies
            dependencies = self._analyze_dependencies(symbols, original_latex)

            # Check for stochastic terms
            is_stochastic = self._is_stochastic(original_latex)

            # Extract differential terms
            differential_terms = self._extract_differential_terms(original_latex)

            return ParsedExpression(
                original_latex=original_latex,
                structure_type=structure_type,
                symbols=symbols,
                operators=operators,
                functions=functions,
                parameters=parameters,
                variables=variables,
                dependencies=dependencies,
                is_stochastic=is_stochastic,
                differential_terms=differential_terms
            )

        except Exception as e:
            logger.error(f"LaTeX parsing failed for: {latex_expression}")
            raise ParseError(f"Failed to parse LaTeX expression: {str(e)}")

    def _classify_structure(self, latex: str) -> MathematicalStructure:
        """Classify the mathematical structure type."""
        # Check for differential equations
        if any(term in latex for term in ['d', '\\partial', '\\frac{d', '\\frac{\\partial']):
            return MathematicalStructure.DIFFERENTIAL

        # Check for integrals
        if '\\int' in latex:
            return MathematicalStructure.INTEGRAL

        # Check for summations
        if '\\sum' in latex:
            return MathematicalStructure.SUMMATION

        # Check for matrices
        if any(env in latex for env in ['\\begin{matrix}', '\\begin{pmatrix}', '\\begin{bmatrix}']):
            return MathematicalStructure.MATRIX

        # Check for inequalities
        if any(op in latex for op in ['<', '>', '\\leq', '\\geq', '\\neq']):
            return MathematicalStructure.INEQUALITY

        # Check for equations
        if '=' in latex:
            return MathematicalStructure.EQUATION

        # Default to expression
        return MathematicalStructure.EXPRESSION

    def _extract_symbols(self, latex: str) -> List[ParsedSymbol]:
        """Extract all mathematical symbols with metadata."""
        symbols = []

        # Find all symbol candidates
        matches = self.symbol_pattern.findall(latex)

        for match in matches:
            symbol_latex = match

            # Extract subscripts
            subscripts = self.subscript_pattern.findall(symbol_latex)

            # Extract superscripts
            superscripts = self.superscript_pattern.findall(symbol_latex)

            # Clean symbol (remove subscripts/superscripts for base symbol)
            clean_symbol = re.sub(r'[_^]\{[^}]+\}', '', symbol_latex)
            clean_symbol = clean_symbol.lstrip('\\')

            # Determine symbol type
            symbol_type = self._determine_symbol_type(clean_symbol)

            # Extract modifiers (bar, hat, tilde, etc.)
            modifiers = self._extract_modifiers(symbol_latex)

            parsed_symbol = ParsedSymbol(
                symbol=clean_symbol,
                latex_form=symbol_latex,
                symbol_type=symbol_type,
                subscripts=subscripts,
                superscripts=superscripts,
                modifiers=modifiers
            )

            symbols.append(parsed_symbol)

        return symbols

    def _determine_symbol_type(self, symbol: str) -> str:
        """Determine the type of a mathematical symbol."""
        # Check if it's a Greek letter (likely parameter)
        if symbol in self.greek_letters:
            return "parameter"

        # Check if it's a mathematical function
        if symbol in self.math_functions:
            return "function"

        # Check if it's an operator
        if symbol in {'+', '-', '*', '/', '=', '<', '>', 'times', 'cdot'}:
            return "operator"

        # Single letters often variables, multi-letter often parameters
        if len(symbol) == 1 and symbol.isalpha():
            return "variable"
        elif len(symbol) > 1:
            return "parameter"

        return "unknown"

    def _extract_modifiers(self, symbol_latex: str) -> List[str]:
        """Extract mathematical modifiers like bar, hat, tilde."""
        modifiers = []

        modifier_patterns = {
            'bar': r'\\bar\{',
            'hat': r'\\hat\{',
            'tilde': r'\\tilde\{',
            'dot': r'\\dot\{',
            'ddot': r'\\ddot\{',
            'vec': r'\\vec\{',
            'overline': r'\\overline\{',
            'underline': r'\\underline\{'
        }

        for modifier, pattern in modifier_patterns.items():
            if re.search(pattern, symbol_latex):
                modifiers.append(modifier)

        return modifiers

    def _extract_operators(self, latex: str) -> List[str]:
        """Extract mathematical operators."""
        operators = []

        # Common operators
        operator_patterns = [
            r'\+', r'-', r'=', r'<', r'>', r'\\leq', r'\\geq', r'\\neq',
            r'\\times', r'\\cdot', r'\\div', r'\\pm', r'\\mp',
            r'\\in', r'\\notin', r'\\subset', r'\\supset',
            r'\\cap', r'\\cup', r'\\setminus'
        ]

        for pattern in operator_patterns:
            matches = re.findall(pattern, latex)
            operators.extend(matches)

        return list(set(operators))  # Remove duplicates

    def _extract_functions(self, latex: str) -> List[str]:
        """Extract mathematical functions."""
        functions = []

        # Look for known mathematical functions
        for func in self.math_functions:
            pattern = f'\\\\{func}\\b'
            if re.search(pattern, latex):
                functions.append(func)

        # Look for custom functions (capital letters followed by parentheses)
        custom_func_pattern = r'([A-Z][a-zA-Z]*)\s*\('
        custom_functions = re.findall(custom_func_pattern, latex)
        functions.extend(custom_functions)

        return list(set(functions))

    def _classify_symbols(self, symbols: List[ParsedSymbol]) -> Tuple[List[str], List[str]]:
        """Classify symbols into parameters and variables."""
        parameters = []
        variables = []

        for symbol in symbols:
            if symbol.symbol_type == "parameter":
                parameters.append(symbol.latex_form)
            elif symbol.symbol_type == "variable":
                variables.append(symbol.latex_form)

        return parameters, variables

    def _analyze_dependencies(self, symbols: List[ParsedSymbol], latex: str) -> Dict[str, List[str]]:
        """Analyze variable dependencies in the expression."""
        dependencies = {}

        # Simple dependency analysis based on position and structure
        # This could be enhanced with more sophisticated parsing

        for symbol in symbols:
            if symbol.symbol_type == "variable":
                deps = []

                # Check if variable depends on time (common in differential equations)
                if any(t_var in latex for t_var in ['t', 'time', '\\tau']):
                    deps.append('t')

                # Add subscript dependencies
                deps.extend(symbol.subscripts)

                dependencies[symbol.symbol] = deps

        return dependencies

    def _is_stochastic(self, latex: str) -> bool:
        """Check if the expression contains stochastic terms."""
        return any(term in latex for term in self.stochastic_indicators)

    def _extract_differential_terms(self, latex: str) -> List[str]:
        """Extract differential terms (dx, dt, dW, etc.)."""
        differential_terms = []

        # Regular differentials
        d_matches = self.differential_pattern.findall(latex)
        differential_terms.extend([f'd{match}' for match in d_matches])

        # Partial differentials
        partial_matches = self.partial_pattern.findall(latex)
        differential_terms.extend([f'\\partial {match}' for match in partial_matches])

        return list(set(differential_terms))

    def extract_equations(self, latex_document: str) -> List[str]:
        """Extract equation blocks from a LaTeX document."""
        equations = []

        # Equation environments
        equation_patterns = [
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\\begin\{align\}(.*?)\\end\{align\}',
            r'\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}',
            r'\$\$(.*?)\$\$',
            r'\$(.*?)\$'
        ]

        for pattern in equation_patterns:
            matches = re.findall(pattern, latex_document, re.DOTALL)
            equations.extend([match.strip() for match in matches])

        return equations

    def validate_mathematical_consistency(self, parsed_expr: ParsedExpression) -> bool:
        """Validate mathematical consistency of parsed expression."""
        # CONSTITUTIONAL REQUIREMENT: Preserve exact mathematical form
        if not parsed_expr.original_latex:
            return False

        # Check that all symbols were properly extracted
        if not parsed_expr.symbols and any(c.isalpha() for c in parsed_expr.original_latex):
            return False

        # Validate differential equation structure
        if parsed_expr.structure_type == MathematicalStructure.DIFFERENTIAL:
            if not parsed_expr.differential_terms:
                return False

        return True

    def to_sympy_expression(self, latex_expression: str) -> Optional[Any]:
        """Convert LaTeX to SymPy expression (requires sympy installation).

        Args:
            latex_expression: LaTeX mathematical expression

        Returns:
            SymPy expression or None if conversion fails
        """
        try:
            import sympy as sp
            from sympy.parsing.latex import parse_latex

            # Attempt to parse with SymPy's LaTeX parser
            sympy_expr = parse_latex(latex_expression)
            return sympy_expr

        except ImportError:
            logger.warning("SymPy not available for LaTeX to SymPy conversion")
            return None
        except Exception as e:
            logger.error(f"Failed to convert LaTeX to SymPy: {str(e)}")
            return None

    def validate_constitutional_compliance(self, parsed_expr: ParsedExpression) -> bool:
        """Validate constitutional mathematical fidelity requirements."""
        # CONSTITUTIONAL REQUIREMENT: Original LaTeX preserved exactly
        if not parsed_expr.original_latex:
            return False

        # CONSTITUTIONAL REQUIREMENT: Mathematical structure preserved
        if not self.validate_mathematical_consistency(parsed_expr):
            return False

        # CONSTITUTIONAL REQUIREMENT: Symbol extraction completeness
        if self.strict_mode:
            # In strict mode, ensure comprehensive symbol extraction
            if parsed_expr.symbols or parsed_expr.variables or parsed_expr.parameters:
                return len(parsed_expr.symbols) > 0

        return True