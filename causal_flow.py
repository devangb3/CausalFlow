"""
CausalFlow: Main orchestrator for the complete framework.

This module integrates all components of the CausalFlow framework:
- Trace Extraction
- Causal Graph Construction
- Causal Attribution
- Counterfactual Repair
- Multi-Agent Critique
"""

from typing import Dict, Any, Optional, List
from trace_logger import TraceLogger
from causal_graph import CausalGraph
from causal_attribution import CausalAttribution
from counterfactual_repair import CounterfactualRepair
from multi_agent_critique import MultiAgentCritique
from llm_client import LLMClient, MultiAgentLLM
import json


class CausalFlow:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-5.1",
        num_critique_agents: int = 3 #Number of agents for multi-agent critique
    ):

        self.llm_client = LLMClient(api_key=api_key, model=model)
        self.multi_agent_llm = MultiAgentLLM(
            num_agents=num_critique_agents,
            api_key=api_key
        )

        self.trace: Optional[TraceLogger] = None
        self.causal_graph: Optional[CausalGraph] = None
        self.causal_attribution: Optional[CausalAttribution] = None
        self.counterfactual_repair: Optional[CounterfactualRepair] = None
        self.multi_agent_critique: Optional[MultiAgentCritique] = None

    def analyze_trace(
        self,
        trace: TraceLogger,
        skip_repair: bool = False,
    ) -> Dict[str, Any]:
        self.trace = trace

        print("CausalFlow Analysis Started")
        print("=" * 60)

        print("\n[1/5] Constructing causal graph...")
        self.causal_graph = CausalGraph(trace)
        print(f"Graph constructed: {self.causal_graph}")

        print("\n[2/5] Performing causal attribution...")
        self.causal_attribution = CausalAttribution(
            trace=trace,
            causal_graph=self.causal_graph,
            llm_client=self.llm_client
        )
        crs_scores = self.causal_attribution.compute_causal_responsibility()
        causal_steps = self.causal_attribution.get_causal_steps()
        print(f"Attribution complete: {len(causal_steps)} causal steps identified")

        if not skip_repair:
            print("\n[3/5] Generating counterfactual repairs...")
            self.counterfactual_repair = CounterfactualRepair(
                trace=trace,
                causal_attribution=self.causal_attribution,
                llm_client=self.llm_client
            )
            repairs = self.counterfactual_repair.generate_repairs()
            print(f"Repair complete: {sum(len(r) for r in repairs.values())} repairs proposed")
        else:
            print("\n[3/5] Skipping counterfactual repair...")
            repairs = {}

        print("\n[4/5] Running multi-agent critique...")
        self.multi_agent_critique = MultiAgentCritique(
            trace=trace,
            causal_attribution=self.causal_attribution,
            multi_agent_llm=self.multi_agent_llm
        )
        critiques = self.multi_agent_critique.critique_causal_attributions()
        consensus_steps = self.multi_agent_critique.get_consensus_causal_steps()
        print(f"Critique complete: {len(consensus_steps)} steps confirmed by consensus")
        

        print("\n[5/5] Compiling results...")
        results = self._compile_results(
            crs_scores,
            causal_steps,
            repairs if not skip_repair else {},
            critiques,
            consensus_steps
        )
        print("Analysis complete!")

        print("\n" + "=" * 60)
        print("Analysis Summary:")
        print(f"  - Causal steps: {len(causal_steps)}")
        print(f"  - Consensus steps: {len(consensus_steps)}")
        print(f"  - Repair proposals: {sum(len(r) for r in repairs.values())}")
        print("=" * 60)

        return results

    def _compile_results(
        self,
        crs_scores: Dict[int, float],
        causal_steps: List[int],
        repairs: Dict[int, List],
        critiques: Dict[int, Any],
        consensus_steps: List[int]
    ) -> Dict[str, Any]:

        results = {
            "trace_summary": {
                "total_steps": len(self.trace.steps),
                "success": self.trace.success,
                "final_answer": self.trace.final_answer,
                "gold_answer": self.trace.gold_answer
            },
            "causal_graph": {
                "statistics": self.causal_graph.get_statistics()
            },
            "causal_attribution": {
                "crs_scores": crs_scores,
                "causal_steps": causal_steps,
                "top_causal_steps": self.causal_attribution.get_top_causal_steps(5)
            },
            "counterfactual_repair": {},
            "multi_agent_critique": {}
        }

        # Add repair results if available
        if self.counterfactual_repair:
            best_repairs = self.counterfactual_repair.get_all_best_repairs()
            results["counterfactual_repair"] = {
                "num_steps_repaired": len(repairs),
                "num_successful_repairs": len(best_repairs),
                "best_repairs": {
                    step_id: {
                        "minimality_score": repair.minimality_score,
                        "success_predicted": repair.success_predicted
                    }
                    for step_id, repair in best_repairs.items()
                }
            }

        # Add critique results if available
        if self.multi_agent_critique:
            results["multi_agent_critique"] = {
                "num_steps_critiqued": len(critiques),
                "consensus_steps": consensus_steps,
                "critique_details": {
                    step_id: {
                        "consensus_score": critique.consensus_score,
                        "final_verdict": critique.final_verdict,
                        "num_critiques": len(critique.critiques)
                    }
                    for step_id, critique in critiques.items()
                }
            }

        return results

    def generate_full_report(self, output_file: Optional[str] = None) -> str:
        sections = []

        sections.append("=" * 70)
        sections.append("CAUSALFLOW COMPREHENSIVE ANALYSIS REPORT")
        sections.append("=" * 70)
        sections.append("")

        # Trace Summary
        sections.append(self._generate_trace_summary())

        # Causal Graph
        if self.causal_graph:
            sections.append("\n" + "=" * 70)
            sections.append("CAUSAL GRAPH")
            sections.append("=" * 70)
            sections.append(str(self.causal_graph.get_statistics()))

        # Causal Attribution
        if self.causal_attribution:
            sections.append("\n" + self.causal_attribution.generate_report())

        # Counterfactual Repair
        if self.counterfactual_repair:
            sections.append("\n" + self.counterfactual_repair.generate_report())

        # Multi-Agent Critique
        if self.multi_agent_critique:
            sections.append("\n" + self.multi_agent_critique.generate_report())

        report = "\n".join(sections)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_file}")

        return report

    def _generate_trace_summary(self) -> str:
        lines = ["TRACE SUMMARY"]
        lines.append("-" * 70)
        lines.append(f"Total Steps: {len(self.trace.steps)}")
        lines.append(f"Outcome: {'SUCCESS' if self.trace.success else 'FAILURE'}")
        lines.append(f"Final Answer: {self.trace.final_answer}")
        lines.append(f"Gold Answer: {self.trace.gold_answer}")
        lines.append("")

        return "\n".join(lines)

    def export_results(self, filepath: str):

        if not self.causal_attribution:
            raise ValueError("No analysis has been performed yet. Call analyze_trace() first.")

        results = self._compile_results(
            self.causal_attribution.crs_scores,
            self.causal_attribution.get_causal_steps(),
            self.counterfactual_repair.repairs if self.counterfactual_repair else {},
            self.multi_agent_critique.critique_results if self.multi_agent_critique else {},
            self.multi_agent_critique.get_consensus_causal_steps() if self.multi_agent_critique else []
        )

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results exported to: {filepath}")