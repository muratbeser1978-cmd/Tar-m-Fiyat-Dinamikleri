"""Dependency Graph model with NetworkX integration and mathematical analysis.

This module implements the DependencyGraph model for analyzing mathematical
variable and equation dependencies with topological ordering capabilities.
"""

from typing import List, Tuple, Optional, Set, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DependencyGraph(BaseModel):
    """Mathematical dependency graph with NetworkX integration.

    This model enforces constitutional mathematical fidelity requirements:
    - Exact preservation of mathematical symbol dependencies
    - Topological ordering for evaluation sequences
    - Parallel execution group identification
    """

    nodes: List[str] = Field(
        ...,
        description="List of mathematical symbols (nodes in the graph)"
    )

    edges: List[List[str]] = Field(
        ...,
        description="List of dependency edges [source, target]"
    )

    evaluation_order: List[str] = Field(
        ...,
        description="Topologically sorted evaluation order"
    )

    cycles: Optional[List[List[str]]] = Field(
        None,
        description="Detected cycles in the dependency graph"
    )

    model_config = ConfigDict(
        # Ensure exact preservation of mathematical symbols
        validate_assignment=True,
        # Allow complex graph structures
        arbitrary_types_allowed=True,
        # Preserve exact mathematical notation
        str_strip_whitespace=False
    )

    @field_validator('edges')
    @classmethod
    def validate_edges_structure(cls, v):
        """Validate edge structure for dependency relationships."""
        if not isinstance(v, list):
            raise ValueError("Edges must be a list")

        for edge in v:
            if not isinstance(edge, list):
                raise ValueError("Each edge must be a list")

            if len(edge) != 2:
                raise ValueError("Each edge must have exactly 2 elements [source, target]")

            if not all(isinstance(node, str) for node in edge):
                raise ValueError("Edge nodes must be strings (mathematical symbols)")

        return v

    @field_validator('evaluation_order')
    @classmethod
    def validate_evaluation_order(cls, v):
        """Validate evaluation order structure."""
        if not isinstance(v, list):
            raise ValueError("Evaluation order must be a list")

        # Allow empty evaluation order for cyclic graphs
        for node in v:
            if not isinstance(node, str):
                raise ValueError("All nodes in evaluation order must be strings")

        return v

    @field_validator('cycles')
    @classmethod
    def validate_cycles(cls, v):
        """Validate cycles structure."""
        if v is None:
            return v

        if not isinstance(v, list):
            raise ValueError("Cycles must be a list")

        for cycle in v:
            if not isinstance(cycle, list):
                raise ValueError("Each cycle must be a list")

            for node in cycle:
                if not isinstance(node, str):
                    raise ValueError("Cycle nodes must be strings")

        return v

    def __str__(self) -> str:
        """String representation showing graph size."""
        return f"DependencyGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"DependencyGraph("
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)}, "
            f"cycles={len(self.cycles) if self.cycles else 0}"
            f")"
        )

    def get_node_count(self) -> int:
        """Get number of nodes in the graph."""
        return len(self.nodes)

    def get_edge_count(self) -> int:
        """Get number of edges in the graph."""
        return len(self.edges)

    def has_cycles(self) -> bool:
        """Check if the graph contains cycles."""
        return self.cycles is not None and len(self.cycles) > 0

    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic (DAG)."""
        return not self.has_cycles()

    def get_dependencies(self, node: str) -> List[str]:
        """Get all nodes that this node depends on."""
        dependencies = []
        for edge in self.edges:
            if edge[1] == node:  # edge[1] is target, edge[0] is source
                dependencies.append(edge[0])
        return dependencies

    def get_dependents(self, node: str) -> List[str]:
        """Get all nodes that depend on this node."""
        dependents = []
        for edge in self.edges:
            if edge[0] == node:  # edge[0] is source, edge[1] is target
                dependents.append(edge[1])
        return dependents

    def get_parallel_groups(self) -> List[List[str]]:
        """Identify groups of nodes that can be computed in parallel."""
        if self.has_cycles():
            return []  # Cannot parallelize with cycles

        # Simple parallel grouping based on evaluation order
        parallel_groups = []
        current_group = []

        for node in self.evaluation_order:
            node_deps = self.get_dependencies(node)

            # Check if all dependencies are satisfied by previous groups
            deps_satisfied = all(
                dep in [n for group in parallel_groups for n in group] or dep == node
                for dep in node_deps
            )

            if deps_satisfied and not any(dep in current_group for dep in node_deps):
                current_group.append(node)
            else:
                if current_group:
                    parallel_groups.append(current_group)
                current_group = [node]

        if current_group:
            parallel_groups.append(current_group)

        return parallel_groups

    def validate_evaluation_order_consistency(self) -> bool:
        """Validate that evaluation order respects dependencies."""
        if self.has_cycles():
            return len(self.evaluation_order) == 0  # Cyclic graphs shouldn't have evaluation order

        # Check that all nodes appear in evaluation order
        if set(self.nodes) != set(self.evaluation_order):
            return False

        # Check that dependencies are satisfied in order
        processed = set()
        for node in self.evaluation_order:
            dependencies = self.get_dependencies(node)
            if not all(dep in processed for dep in dependencies):
                return False
            processed.add(node)

        return True

    def get_root_nodes(self) -> List[str]:
        """Get nodes with no dependencies (root nodes)."""
        all_targets = {edge[1] for edge in self.edges}
        return [node for node in self.nodes if node not in all_targets]

    def get_leaf_nodes(self) -> List[str]:
        """Get nodes with no dependents (leaf nodes)."""
        all_sources = {edge[0] for edge in self.edges}
        return [node for node in self.nodes if node not in all_sources]

    def compute_topological_order(self) -> List[str]:
        """Compute topological ordering using Kahn's algorithm."""
        if self.has_cycles():
            return []

        # Implementation of Kahn's algorithm
        in_degree = {node: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge[1]] += 1

        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for dependent in self.get_dependents(node):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result if len(result) == len(self.nodes) else []

    def find_cycles(self) -> List[List[str]]:
        """Find all cycles in the graph using DFS."""
        def dfs_visit(node: str, visited: Set[str], rec_stack: Set[str], path: List[str]) -> List[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            cycles_found = []

            for dependent in self.get_dependents(node):
                if dependent in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(dependent)
                    cycle = path[cycle_start:] + [dependent]
                    cycles_found.append(cycle)
                elif dependent not in visited:
                    cycles_found.extend(dfs_visit(dependent, visited, rec_stack, path.copy()))

            rec_stack.remove(node)
            return cycles_found

        visited = set()
        all_cycles = []

        for node in self.nodes:
            if node not in visited:
                cycles = dfs_visit(node, visited, set(), [])
                all_cycles.extend(cycles)

        return all_cycles

    def to_networkx(self):
        """Convert to NetworkX graph for advanced analysis."""
        try:
            import networkx as nx

            G = nx.DiGraph()
            G.add_nodes_from(self.nodes)
            G.add_edges_from(self.edges)
            return G
        except ImportError:
            raise ImportError("NetworkX is required for graph analysis")

    def validate_mathematical_consistency(self) -> bool:
        """Validate mathematical consistency of the dependency graph."""
        # Check that all edge nodes exist in the node list
        edge_nodes = set()
        for edge in self.edges:
            edge_nodes.update(edge)

        if not edge_nodes.issubset(set(self.nodes)):
            return False

        # Check evaluation order consistency
        if not self.validate_evaluation_order_consistency():
            return False

        return True

    def get_critical_path(self) -> List[str]:
        """Get the critical path (longest dependency chain)."""
        if self.has_cycles():
            return []

        # Simple critical path using evaluation order
        return self.evaluation_order.copy()

    def analyze_complexity(self) -> Dict[str, int]:
        """Analyze graph complexity metrics."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_cycles": len(self.cycles) if self.cycles else 0,
            "max_in_degree": max(len(self.get_dependencies(node)) for node in self.nodes) if self.nodes else 0,
            "max_out_degree": max(len(self.get_dependents(node)) for node in self.nodes) if self.nodes else 0,
            "num_root_nodes": len(self.get_root_nodes()),
            "num_leaf_nodes": len(self.get_leaf_nodes())
        }

    def validate_constitutional_compliance(self) -> bool:
        """Validate constitutional mathematical fidelity requirements."""
        # CONSTITUTIONAL REQUIREMENT: Mathematical symbols preserved exactly
        for node in self.nodes:
            if not isinstance(node, str) or not node:
                return False

        # CONSTITUTIONAL REQUIREMENT: Dependency relationships preserved exactly
        if not self.validate_mathematical_consistency():
            return False

        return True