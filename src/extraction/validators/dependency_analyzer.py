"""Variable Dependency Analyzer for mathematical expressions.

This module implements the DependencyAnalyzer class with constitutional
mathematical fidelity requirements for analyzing variable dependencies
and mathematical relationships.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re
import logging
from collections import defaultdict, deque

from ..parsers.latex_parser import LaTeXParser, ParsedExpression
from ..extractors.equation_extractor import ExtractedEquation, EquationType
from ...core.equations.dependency_graph import DependencyGraph
from ...core.variables.mathematical_variable import MathematicalVariable, VariableType

# Configure logging
logger = logging.getLogger(__name__)


class DependencyError(Exception):
    """Exception raised when dependency analysis fails."""
    pass


class DependencyType(str, Enum):
    """Types of mathematical dependencies."""
    DIRECT = "DIRECT"              # Direct functional dependence: y = f(x)
    DIFFERENTIAL = "DIFFERENTIAL"  # Differential dependence: dy/dx
    STOCHASTIC = "STOCHASTIC"     # Stochastic dependence: dX = μ dt + σ dW
    TEMPORAL = "TEMPORAL"         # Time dependence: X(t)
    CONDITIONAL = "CONDITIONAL"   # Conditional dependence: X|Y
    PARAMETRIC = "PARAMETRIC"     # Parameter dependence: f(x; θ)
    IMPLICIT = "IMPLICIT"         # Implicit dependence: F(x,y) = 0


class VariableRole(str, Enum):
    """Role of variable in mathematical context."""
    DEPENDENT = "DEPENDENT"        # Left-hand side variable
    INDEPENDENT = "INDEPENDENT"    # Right-hand side variable
    STATE = "STATE"               # State variable in dynamic system
    CONTROL = "CONTROL"           # Control variable
    PARAMETER = "PARAMETER"       # Model parameter
    NOISE = "NOISE"               # Stochastic noise term
    OBSERVABLE = "OBSERVABLE"     # Observable variable
    LATENT = "LATENT"            # Latent/unobserved variable

def convert_role_to_variable_type(role: VariableRole) -> VariableType:
    """Convert VariableRole to VariableType for MathematicalVariable."""
    role_mapping = {
        VariableRole.STATE: VariableType.STATE,
        VariableRole.DEPENDENT: VariableType.ALGEBRAIC,
        VariableRole.INDEPENDENT: VariableType.ALGEBRAIC,
        VariableRole.PARAMETER: VariableType.PARAMETER,
        VariableRole.CONTROL: VariableType.PARAMETER,
        VariableRole.NOISE: VariableType.ALGEBRAIC,
        VariableRole.OBSERVABLE: VariableType.ALGEBRAIC,
        VariableRole.LATENT: VariableType.ALGEBRAIC
    }
    return role_mapping.get(role, VariableType.ALGEBRAIC)


@dataclass
class VariableDependency:
    """Represents a dependency relationship between variables."""
    source_variable: str
    target_variable: str
    dependency_type: DependencyType
    equation_id: str
    strength: float  # 0.0 to 1.0
    context: str
    is_causal: bool
    time_lag: Optional[int]


@dataclass
class VariableNode:
    """Node representing a variable in the dependency graph."""
    symbol: str
    latex_form: str
    role: VariableRole
    data_type: str  # scalar, vector, matrix, function
    domain: Optional[str]
    units: Optional[str]
    equation_ids: List[str]
    dependencies: List[str]
    dependents: List[str]


@dataclass
class DependencyAnalysisResult:
    """Result of comprehensive dependency analysis."""
    variables: Dict[str, VariableNode]
    dependencies: List[VariableDependency]
    dependency_graph: DependencyGraph
    strongly_connected_components: List[List[str]]
    causal_ordering: List[str]
    cycles: List[List[str]]
    temporal_structure: Dict[str, List[str]]
    stochastic_structure: Dict[str, List[str]]
    validation_errors: List[str]


class DependencyAnalyzer:
    """Variable dependency analyzer with mathematical structure analysis.

    This class enforces constitutional mathematical fidelity requirements:
    - Exact preservation of variable symbols and mathematical relationships
    - Comprehensive dependency classification and analysis
    - Mathematical structure validation and causal inference
    """

    def __init__(self, latex_parser: Optional[LaTeXParser] = None):
        """Initialize dependency analyzer.

        Args:
            latex_parser: LaTeX parser instance (creates new if None)
        """
        self.latex_parser = latex_parser or LaTeXParser(strict_mode=True)
        self._initialize_patterns()
        self._initialize_analyzers()

    def _initialize_patterns(self):
        """Initialize regex patterns for dependency analysis."""
        # Dependency relationship patterns
        self.assignment_pattern = re.compile(r'([^=]+)=([^=]+)')
        self.differential_pattern = re.compile(r'd([a-zA-Z_]+(?:_\{[^}]+\})?)')
        self.partial_pattern = re.compile(r'\\frac\{\\partial\s*([^}]+)\}\{\\partial\s*([^}]+)\}')
        self.function_pattern = re.compile(r'([a-zA-Z_]+(?:_\{[^}]+\})?)\s*\(([^)]+)\)')
        self.conditional_pattern = re.compile(r'([^|]+)\|([^|]+)')

        # Variable role patterns
        self.state_indicators = ['X', 'Y', 'Z', 'S', 'x', 'y', 'z', 's']
        self.parameter_indicators = ['\\alpha', '\\beta', '\\gamma', '\\theta', '\\mu', '\\sigma', '\\kappa']
        self.noise_indicators = ['W', 'B', 'N', 'epsilon', '\\epsilon', '\\xi', '\\eta']

        # Time dependence patterns
        self.time_pattern = re.compile(r'([a-zA-Z_]+(?:_\{[^}]+\})?)\s*\(.*?t.*?\)')
        self.time_subscript_pattern = re.compile(r'([a-zA-Z_]+)_\{([^}]*t[^}]*)\}')

    def _initialize_analyzers(self):
        """Initialize specialized analyzers."""
        self.causal_keywords = {
            'causes', 'leads to', 'results in', 'implies', 'determines',
            'influences', 'affects', 'drives', 'generates'
        }

        self.temporal_keywords = {
            'time', 'period', 'lag', 'delay', 'future', 'past', 'current'
        }

    def analyze_dependencies(self, equations: List[ExtractedEquation]) -> DependencyAnalysisResult:
        """Perform comprehensive dependency analysis on extracted equations.

        Args:
            equations: List of extracted equations to analyze

        Returns:
            DependencyAnalysisResult: Comprehensive analysis results

        Raises:
            DependencyError: If analysis fails critically
        """
        try:
            # Initialize analysis structures
            variables = {}
            dependencies = []
            validation_errors = []

            # Extract all variables and their roles
            variables = self._extract_variables(equations)

            # Analyze dependencies between variables
            dependencies = self._analyze_equation_dependencies(equations, variables)

            # Build dependency graph
            dependency_graph = self._build_dependency_graph(variables, dependencies)

            # Analyze graph structure
            strongly_connected_components = self._find_strongly_connected_components(dependency_graph)
            cycles = self._detect_cycles(dependency_graph)
            causal_ordering = self._determine_causal_ordering(dependency_graph)

            # Analyze temporal and stochastic structure
            temporal_structure = self._analyze_temporal_structure(equations, variables)
            stochastic_structure = self._analyze_stochastic_structure(equations, variables)

            # Validate analysis results
            validation_errors = self._validate_analysis(variables, dependencies, dependency_graph)

            return DependencyAnalysisResult(
                variables=variables,
                dependencies=dependencies,
                dependency_graph=dependency_graph,
                strongly_connected_components=strongly_connected_components,
                causal_ordering=causal_ordering,
                cycles=cycles,
                temporal_structure=temporal_structure,
                stochastic_structure=stochastic_structure,
                validation_errors=validation_errors
            )

        except Exception as e:
            logger.error(f"Dependency analysis failed: {str(e)}")
            raise DependencyError(f"Failed to analyze dependencies: {str(e)}")

    def _extract_variables(self, equations: List[ExtractedEquation]) -> Dict[str, VariableNode]:
        """Extract all variables and classify their roles."""
        variables = {}

        for equation in equations:
            # Parse equation to get detailed variable information
            try:
                parsed = self.latex_parser.parse_latex(equation.latex_form)

                # Process all symbols
                all_symbols = parsed.variables + parsed.parameters

                for symbol in all_symbols:
                    if symbol not in variables:
                        # Determine variable role
                        role = self._determine_variable_role(symbol, equation, parsed)

                        # Determine data type
                        data_type = self._determine_data_type(symbol, equation.latex_form)

                        variables[symbol] = VariableNode(
                            symbol=symbol,
                            latex_form=symbol,
                            role=role,
                            data_type=data_type,
                            domain=None,  # Could be enhanced
                            units=None,   # Could be enhanced
                            equation_ids=[equation.equation_id],
                            dependencies=[],
                            dependents=[]
                        )
                    else:
                        # Add equation reference
                        if equation.equation_id not in variables[symbol].equation_ids:
                            variables[symbol].equation_ids.append(equation.equation_id)

            except Exception as e:
                logger.warning(f"Failed to parse equation {equation.equation_id}: {str(e)}")

        return variables

    def _determine_variable_role(self, symbol: str, equation: ExtractedEquation, parsed: ParsedExpression) -> VariableRole:
        """Determine the role of a variable in the mathematical context."""
        latex = equation.latex_form

        # Check if it's a parameter (Greek letters, etc.)
        clean_symbol = symbol.replace('\\', '').replace('{', '').replace('}', '')
        if any(param in symbol for param in self.parameter_indicators):
            return VariableRole.PARAMETER

        # Check if it's a noise term
        if any(noise in symbol for noise in self.noise_indicators):
            return VariableRole.NOISE

        # Check if it appears on left side of assignment (dependent variable)
        assignment_match = self.assignment_pattern.search(latex)
        if assignment_match:
            left_side = assignment_match.group(1).strip()
            if symbol in left_side or any(var in left_side for var in [symbol]):
                return VariableRole.DEPENDENT

        # Check for differential terms (state variables)
        if equation.equation_type == EquationType.SDE or equation.equation_type == EquationType.ODE:
            if f'd{symbol}' in latex or f'd{clean_symbol}' in latex:
                return VariableRole.STATE

        # Check for control variable indicators
        if any(control in symbol.lower() for control in ['u', 'control', 'policy']):
            return VariableRole.CONTROL

        # Default classification
        if clean_symbol in self.state_indicators:
            return VariableRole.STATE
        elif symbol in parsed.parameters:
            return VariableRole.PARAMETER
        else:
            return VariableRole.INDEPENDENT

    def _determine_data_type(self, symbol: str, latex: str) -> str:
        """Determine the data type of a variable."""
        # Check for vector notation
        if '\\vec{' in symbol or '\\mathbf{' in symbol:
            return "vector"

        # Check for matrix notation
        if '\\mathbf{' in symbol and symbol.isupper():
            return "matrix"

        # Check for function notation
        if re.search(rf'{re.escape(symbol)}\s*\(', latex):
            return "function"

        # Default to scalar
        return "scalar"

    def _analyze_equation_dependencies(self, equations: List[ExtractedEquation],
                                     variables: Dict[str, VariableNode]) -> List[VariableDependency]:
        """Analyze dependencies within and between equations."""
        dependencies = []

        for equation in equations:
            try:
                parsed = self.latex_parser.parse_latex(equation.latex_form)
                equation_deps = self._extract_equation_dependencies(equation, parsed, variables)
                dependencies.extend(equation_deps)
            except Exception as e:
                logger.warning(f"Failed to analyze dependencies in equation {equation.equation_id}: {str(e)}")

        return dependencies

    def _extract_equation_dependencies(self, equation: ExtractedEquation,
                                     parsed: ParsedExpression,
                                     variables: Dict[str, VariableNode]) -> List[VariableDependency]:
        """Extract dependencies from a single equation."""
        dependencies = []
        latex = equation.latex_form

        # Assignment dependencies (y = f(x))
        assignment_match = self.assignment_pattern.search(latex)
        if assignment_match:
            left_side = assignment_match.group(1).strip()
            right_side = assignment_match.group(2).strip()

            # Find variables on left and right sides
            left_vars = self._find_variables_in_expression(left_side, variables)
            right_vars = self._find_variables_in_expression(right_side, variables)

            # Create dependencies from right to left
            for left_var in left_vars:
                for right_var in right_vars:
                    if left_var != right_var:
                        dep = VariableDependency(
                            source_variable=right_var,
                            target_variable=left_var,
                            dependency_type=DependencyType.DIRECT,
                            equation_id=equation.equation_id,
                            strength=1.0,
                            context=equation.surrounding_text,
                            is_causal=self._is_causal_relationship(right_var, left_var, equation),
                            time_lag=self._extract_time_lag(latex)
                        )
                        dependencies.append(dep)

        # Differential dependencies
        if equation.equation_type in [EquationType.SDE, EquationType.ODE, EquationType.PDE]:
            diff_deps = self._extract_differential_dependencies(equation, parsed, variables)
            dependencies.extend(diff_deps)

        # Function dependencies
        func_deps = self._extract_function_dependencies(equation, variables)
        dependencies.extend(func_deps)

        return dependencies

    def _find_variables_in_expression(self, expression: str, variables: Dict[str, VariableNode]) -> List[str]:
        """Find all variables mentioned in an expression."""
        found_vars = []
        for var_symbol in variables.keys():
            if var_symbol in expression:
                found_vars.append(var_symbol)
        return found_vars

    def _extract_differential_dependencies(self, equation: ExtractedEquation,
                                         parsed: ParsedExpression,
                                         variables: Dict[str, VariableNode]) -> List[VariableDependency]:
        """Extract dependencies from differential equations."""
        dependencies = []
        latex = equation.latex_form

        # Find differential terms (dx, dy, etc.)
        diff_matches = self.differential_pattern.findall(latex)

        for diff_var in diff_matches:
            # This differential variable depends on other variables in the equation
            other_vars = [v for v in parsed.variables + parsed.parameters
                         if v != diff_var and v in variables]

            for other_var in other_vars:
                dep_type = DependencyType.STOCHASTIC if parsed.is_stochastic else DependencyType.DIFFERENTIAL

                dep = VariableDependency(
                    source_variable=other_var,
                    target_variable=diff_var,
                    dependency_type=dep_type,
                    equation_id=equation.equation_id,
                    strength=0.8,  # High strength for differential relationships
                    context=equation.surrounding_text,
                    is_causal=True,  # Differential relationships are generally causal
                    time_lag=None
                )
                dependencies.append(dep)

        return dependencies

    def _extract_function_dependencies(self, equation: ExtractedEquation,
                                     variables: Dict[str, VariableNode]) -> List[VariableDependency]:
        """Extract dependencies from function relationships."""
        dependencies = []
        latex = equation.latex_form

        # Find function calls
        func_matches = self.function_pattern.findall(latex)

        for func_name, func_args in func_matches:
            if func_name in variables:
                # Function arguments are dependencies
                arg_vars = self._find_variables_in_expression(func_args, variables)

                for arg_var in arg_vars:
                    dep = VariableDependency(
                        source_variable=arg_var,
                        target_variable=func_name,
                        dependency_type=DependencyType.DIRECT,
                        equation_id=equation.equation_id,
                        strength=0.9,
                        context=equation.surrounding_text,
                        is_causal=self._is_causal_relationship(arg_var, func_name, equation),
                        time_lag=None
                    )
                    dependencies.append(dep)

        return dependencies

    def _is_causal_relationship(self, source: str, target: str, equation: ExtractedEquation) -> bool:
        """Determine if a relationship is causal."""
        context = equation.surrounding_text.lower()

        # Check for causal keywords in surrounding text
        if any(keyword in context for keyword in self.causal_keywords):
            return True

        # Differential relationships are generally causal
        if equation.equation_type in [EquationType.SDE, EquationType.ODE]:
            return True

        # Time dependencies are often causal
        if any(keyword in context for keyword in self.temporal_keywords):
            return True

        return False

    def _extract_time_lag(self, latex: str) -> Optional[int]:
        """Extract time lag information from equation."""
        # Look for patterns like X_{t-1}, X_{t+1}, etc.
        lag_pattern = r't\s*([+-])\s*(\d+)'
        matches = re.findall(lag_pattern, latex)

        if matches:
            sign, value = matches[0]
            lag = int(value) if sign == '+' else -int(value)
            return lag

        return None

    def _build_dependency_graph(self, variables: Dict[str, VariableNode],
                               dependencies: List[VariableDependency]) -> DependencyGraph:
        """Build NetworkX dependency graph from variables and dependencies."""
        # Create nodes and edges for dependency graph
        nodes = list(variables.keys())
        edges = [[dep.source_variable, dep.target_variable] for dep in dependencies]

        # Compute evaluation order (topological sort)
        evaluation_order = self._compute_topological_order(nodes, edges)

        # Detect cycles
        cycles = self._detect_cycles_in_edges(nodes, edges)

        return DependencyGraph(
            nodes=nodes,
            edges=edges,
            evaluation_order=evaluation_order,
            cycles=cycles
        )

    def _compute_topological_order(self, nodes: List[str], edges: List[List[str]]) -> List[str]:
        """Compute topological ordering using Kahn's algorithm."""
        # Calculate in-degrees
        in_degree = {node: 0 for node in nodes}
        adj_list = defaultdict(list)

        for source, target in edges:
            if source in in_degree and target in in_degree:
                adj_list[source].append(target)
                in_degree[target] += 1

        # Initialize queue with nodes having no incoming edges
        queue = deque([node for node in nodes if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Remove edges from this node
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Return empty list if cycle detected
        return result if len(result) == len(nodes) else []

    def _detect_cycles_in_edges(self, nodes: List[str], edges: List[List[str]]) -> List[List[str]]:
        """Detect cycles in the dependency graph."""
        adj_list = defaultdict(list)
        for source, target in edges:
            if source in nodes and target in nodes:
                adj_list[source].append(target)

        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adj_list[node]:
                if neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif neighbor not in visited:
                    dfs(neighbor, path.copy())

            rec_stack.remove(node)

        for node in nodes:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _find_strongly_connected_components(self, graph: DependencyGraph) -> List[List[str]]:
        """Find strongly connected components using Tarjan's algorithm."""
        # Simplified implementation - could be enhanced
        components = []

        if graph.has_cycles():
            # If there are cycles, the cyclic nodes form strongly connected components
            for cycle in graph.cycles or []:
                if cycle not in components:
                    components.append(cycle)

        # Individual nodes that are not part of cycles
        cyclic_nodes = set()
        for cycle in graph.cycles or []:
            cyclic_nodes.update(cycle)

        for node in graph.nodes:
            if node not in cyclic_nodes:
                components.append([node])

        return components

    def _detect_cycles(self, graph: DependencyGraph) -> List[List[str]]:
        """Detect cycles in the dependency graph."""
        return graph.cycles or []

    def _determine_causal_ordering(self, graph: DependencyGraph) -> List[str]:
        """Determine causal ordering of variables."""
        # Use topological ordering as causal ordering for acyclic graphs
        if graph.is_acyclic():
            return graph.evaluation_order
        else:
            # For cyclic graphs, try to find a partial ordering
            return []

    def _analyze_temporal_structure(self, equations: List[ExtractedEquation],
                                   variables: Dict[str, VariableNode]) -> Dict[str, List[str]]:
        """Analyze temporal dependencies and structure."""
        temporal_structure = {}

        for equation in equations:
            # Find variables with time subscripts or function arguments
            time_vars = []

            for var_symbol in variables.keys():
                if self.time_pattern.search(var_symbol) or self.time_subscript_pattern.search(var_symbol):
                    time_vars.append(var_symbol)

            if time_vars:
                temporal_structure[equation.equation_id] = time_vars

        return temporal_structure

    def _analyze_stochastic_structure(self, equations: List[ExtractedEquation],
                                     variables: Dict[str, VariableNode]) -> Dict[str, List[str]]:
        """Analyze stochastic dependencies and noise structure."""
        stochastic_structure = {}

        for equation in equations:
            if equation.equation_type == EquationType.SDE:
                # Find stochastic variables (noise terms)
                stochastic_vars = []

                for var_symbol in variables.keys():
                    if variables[var_symbol].role == VariableRole.NOISE:
                        stochastic_vars.append(var_symbol)

                if stochastic_vars:
                    stochastic_structure[equation.equation_id] = stochastic_vars

        return stochastic_structure

    def _validate_analysis(self, variables: Dict[str, VariableNode],
                          dependencies: List[VariableDependency],
                          graph: DependencyGraph) -> List[str]:
        """Validate the dependency analysis results."""
        validation_errors = []

        # Check that all dependency variables exist
        for dep in dependencies:
            if dep.source_variable not in variables:
                validation_errors.append(f"Source variable {dep.source_variable} not found in variables")
            if dep.target_variable not in variables:
                validation_errors.append(f"Target variable {dep.target_variable} not found in variables")

        # Check graph consistency
        if not graph.validate_mathematical_consistency():
            validation_errors.append("Dependency graph is mathematically inconsistent")

        return validation_errors

    def validate_constitutional_compliance(self, result: DependencyAnalysisResult) -> bool:
        """Validate constitutional mathematical fidelity requirements."""
        # CONSTITUTIONAL REQUIREMENT: All variable symbols preserved exactly
        for var_symbol, var_node in result.variables.items():
            if var_node.symbol != var_symbol:
                return False

        # CONSTITUTIONAL REQUIREMENT: Mathematical relationships preserved
        if not result.dependency_graph.validate_constitutional_compliance():
            return False

        return True