"""
CausalGraph: Constructs directed acyclic graphs (DAGs) from agent traces.

This module implements the causal graph construction component of CausalFlow
as specified in Section 4 of the research proposal.
"""

import networkx as nx
from typing import List, Set, Dict, Any, Optional
from trace_logger import TraceLogger, Step, StepType


class CausalGraph:
    """
    Represents the causal structure of an agent's execution trace as a DAG.

    The graph captures dependencies between steps, allowing us to understand
    how information and reasoning flow through the agent, and how errors propagate.
    """

    def __init__(self, trace: TraceLogger):
        """
        Initialize a causal graph from a trace.

        Args:
            trace: The TraceLogger instance containing the execution trace
        """
        self.trace = trace
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """
        Construct the DAG from the trace.

        Process:
        1. Create a node for every step in the trace
        2. For each step, inspect its dependencies field
        3. Add directed edges from dependent steps to current step
        """
        # Add all steps as nodes
        for step in self.trace.steps:
            self.graph.add_node(
                step.step_id,
                step_type=step.step_type.value,
                step=step
            )

        # Add edges based on dependencies
        for step in self.trace.steps:
            for dep_id in step.dependencies:
                # Edge from dependency to current step (dep -> step)
                self.graph.add_edge(dep_id, step.step_id)

    def get_ancestors(self, step_id: int) -> Set[int]:
        """
        Get all ancestor nodes (steps that causally precede) a given step.

        Args:
            step_id: The step ID to find ancestors for

        Returns:
            Set of step IDs that are ancestors
        """
        if step_id not in self.graph:
            return set()
        return nx.ancestors(self.graph, step_id)

    def get_descendants(self, step_id: int) -> Set[int]:
        """
        Get all descendant nodes (steps that causally follow) a given step.

        Args:
            step_id: The step ID to find descendants for

        Returns:
            Set of step IDs that are descendants
        """
        if step_id not in self.graph:
            return set()
        return nx.descendants(self.graph, step_id)

    def get_immediate_dependencies(self, step_id: int) -> List[int]:
        """
        Get immediate dependencies (parents) of a step.

        Args:
            step_id: The step ID

        Returns:
            List of step IDs that this step directly depends on
        """
        if step_id not in self.graph:
            return []
        return list(self.graph.predecessors(step_id))

    def get_immediate_dependents(self, step_id: int) -> List[int]:
        """
        Get immediate dependents (children) of a step.

        Args:
            step_id: The step ID

        Returns:
            List of step IDs that directly depend on this step
        """
        if step_id not in self.graph:
            return []
        return list(self.graph.successors(step_id))

    def is_ancestor(self, ancestor_id: int, descendant_id: int) -> bool:
        """
        Check if one step is an ancestor of another.

        Args:
            ancestor_id: Potential ancestor step ID
            descendant_id: Potential descendant step ID

        Returns:
            True if ancestor_id causally precedes descendant_id
        """
        if ancestor_id not in self.graph or descendant_id not in self.graph:
            return False
        return nx.has_path(self.graph, ancestor_id, descendant_id)

    def get_causal_path(self, from_step: int, to_step: int) -> Optional[List[int]]:
        """
        Find a causal path between two steps.

        Args:
            from_step: Starting step ID
            to_step: Ending step ID

        Returns:
            List of step IDs forming the path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.graph, from_step, to_step)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_all_paths_to_final(self, step_id: int) -> List[List[int]]:
        """
        Get all causal paths from a step to the final answer.

        Args:
            step_id: The starting step ID

        Returns:
            List of paths (each path is a list of step IDs)
        """
        if "FINAL" not in self.graph or step_id not in self.graph:
            return []

        try:
            return list(nx.all_simple_paths(self.graph, step_id, "FINAL"))
        except nx.NodeNotFound:
            return []

    def topological_order(self) -> List[int]:
        """
        Get steps in topological order (respecting causal dependencies).

        Returns:
            List of step IDs in topological order
        """
        # Filter out the special FINAL node
        return [
            node for node in nx.topological_sort(self.graph)
            if node != "FINAL"
        ]

    def get_step_depth(self, step_id: int) -> int:
        """
        Get the depth of a step (longest path from root).

        Args:
            step_id: The step ID

        Returns:
            Depth of the step in the graph
        """
        if step_id not in self.graph:
            return -1

        # Find all nodes with no predecessors (roots)
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

        if not roots:
            return 0

        max_depth = 0
        for root in roots:
            try:
                path = nx.shortest_path(self.graph, root, step_id)
                max_depth = max(max_depth, len(path) - 1)
            except nx.NetworkXNoPath:
                continue

        return max_depth

    def get_critical_steps(self) -> List[int]:
        """
        Identify critical steps that lie on all paths to the final answer.

        Returns:
            List of critical step IDs
        """
        if "FINAL" not in self.graph:
            return []

        # Get all nodes that must be traversed to reach FINAL
        roots = [n for n in self.graph.nodes()
                if n != "FINAL" and self.graph.in_degree(n) == 0]

        if not roots:
            return []

        # Find nodes common to all paths from any root to FINAL
        critical = None
        for root in roots:
            try:
                paths = list(nx.all_simple_paths(self.graph, root, "FINAL"))
                if paths:
                    # Nodes in all paths from this root
                    nodes_in_all = set(paths[0])
                    for path in paths[1:]:
                        nodes_in_all &= set(path)

                    if critical is None:
                        critical = nodes_in_all
                    else:
                        critical |= nodes_in_all
            except nx.NodeNotFound:
                continue

        if critical:
            critical.discard("FINAL")
            return sorted(list(critical))
        return []

    def visualize_graph(self, filename: str = None) -> str:
        """
        Generate a text-based visualization of the graph.

        Args:
            filename: Optional filename to save visualization

        Returns:
            String representation of the graph
        """
        lines = ["Causal Graph Structure:"]
        lines.append("=" * 50)

        for step_id in self.topological_order():
            step = self.trace.get_step(step_id)
            if not step:
                continue

            deps = self.get_immediate_dependencies(step_id)
            deps_str = f" <- {deps}" if deps else ""

            step_info = f"Step {step_id} [{step.step_type.value}]{deps_str}"
            lines.append(step_info)

            # Add step content summary
            if step.text:
                text_preview = step.text[:60] + "..." if len(step.text) > 60 else step.text
                lines.append(f"  Text: {text_preview}")
            if step.tool_name:
                lines.append(f"  Tool: {step.tool_name}")
            if step.action:
                lines.append(f"  Action: {step.action}")

            lines.append("")

        viz_str = "\n".join(lines)

        if filename:
            with open(filename, 'w') as f:
                f.write(viz_str)

        return viz_str

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the causal graph.

        Returns:
            Dictionary containing graph statistics
        """
        total_nodes = len(self.graph.nodes()) - (1 if "FINAL" in self.graph else 0)

        step_types = {}
        for step in self.trace.steps:
            step_type = step.step_type.value
            step_types[step_type] = step_types.get(step_type, 0) + 1

        return {
            "total_steps": total_nodes,
            "total_edges": len(self.graph.edges()),
            "step_types": step_types,
            "max_depth": max([self.get_step_depth(s.step_id) for s in self.trace.steps]) if self.trace.steps else 0,
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "critical_steps": len(self.get_critical_steps())
        }
