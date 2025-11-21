import copy
from typing import Dict, Any, Optional, List
from trace_logger import TraceLogger, Step
from causal_graph import CausalGraph
from causal_attribution import CausalAttribution
from counterfactual_repair import CounterfactualRepair
from multi_agent_critique import MultiAgentCritique
from llm_client import LLMClient, MultiAgentLLM
from mongodb_storage import MongoDBStorage

class CausalFlow:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-flash-lite",
        num_critique_agents: int = 3, #Number of agents for multi-agent critique
        mongo_storage: Optional[MongoDBStorage] = None
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
        self.mongo_storage = mongo_storage

    def analyze_trace(
        self,
        trace: TraceLogger
    ) -> Dict[str, Any]:

        self.trace = trace

        print("\n[1/5] Constructing causal graph")
        self.causal_graph = CausalGraph(self.trace)

        print("\n[2/5] Performing causal attribution")
        self.causal_attribution = CausalAttribution(
            trace=self.trace,
            causal_graph=self.causal_graph,
            llm_client=self.llm_client
        )
        crs_scores = self.causal_attribution.compute_causal_responsibility()
        causal_steps = self.causal_attribution.get_causal_steps()
        print(f"Attribution complete: {len(causal_steps)} causal steps identified")
        
        print("\n[3/5] Generating counterfactual repairs")
        self.counterfactual_repair = CounterfactualRepair(
            trace=self.trace,
            causal_attribution=self.causal_attribution,
            llm_client=self.llm_client
        )
        repairs = self.counterfactual_repair.generate_repairs(step_ids=causal_steps)
        print(f"Repair complete: {sum(len(r) for r in repairs.values())} repairs proposed")
        
        print("\n[4/5] Running multi-agent critique")
        self.multi_agent_critique = MultiAgentCritique(
            trace=self.trace,
            causal_attribution=self.causal_attribution,
            multi_agent_llm=self.multi_agent_llm
        )
        critiques = self.multi_agent_critique.critique_causal_attributions()
        consensus_steps = self.multi_agent_critique.get_consensus_causal_steps()
        print(f"Critique complete: {len(consensus_steps)} steps confirmed by consensus")


        print("\n[5/5] Compiling results")
        results = self._compile_results(
            crs_scores,
            causal_steps,
            repairs,
            critiques,
            consensus_steps
        )

        print(f"\n[6/6] Generating metrics")
        metrics = self.generate_metrics(consensus_steps)
        results['metrics'] = metrics

        return results

    def _compile_results(
        self,
        crs_scores: Dict[int, float],
        causal_steps: List[int],
        repairs: Dict[int, List],
        critiques: Dict[int, Any],
        consensus_steps: List[Step]
    ) -> Dict[str, Any]:

        results = {
            "trace_summary": {
                "total_steps": len(self.trace.steps) if self.trace else 0,
                "success": self.trace.success if self.trace else False,
                "final_answer": self.trace.final_answer if self.trace else "",
                "gold_answer": self.trace.gold_answer if self.trace else ""
            },
            "causal_graph": {
                "statistics": self.causal_graph.get_statistics() if self.causal_graph else {}
            },
            "causal_attribution": {
                "crs_scores": crs_scores if crs_scores else {},
                "causal_steps": causal_steps if causal_steps else []
            },
            "counterfactual_repair": {},
            "multi_agent_critique": {}
        }

        if self.counterfactual_repair:
            best_repairs = self.counterfactual_repair.get_all_best_repairs()
            results["counterfactual_repair"] = {
                "num_steps_repaired": len(repairs) if repairs else 0,
                "num_successful_repairs": len(best_repairs) if best_repairs else 0,
                "best_repairs": {
                    step_id: {
                        "minimality_score": repair.minimality_score if repair else 0.0,
                        "success_predicted": repair.success_predicted if repair else False
                    }
                    for step_id, repair in best_repairs.items()
                }
            }

        # Add critique results if available
        if self.multi_agent_critique:
            results["multi_agent_critique"] = {
                "num_steps_critiqued": len(critiques) if critiques else 0,
                "consensus_steps": [step.to_dict() for step in consensus_steps] if consensus_steps else [],
                "critique_details": {
                    step_id: {
                        "consensus_score": critique.consensus_score if critique else 0.0,
                        "final_verdict": critique.final_verdict if critique else False,
                        "num_critiques": len(critique.critiques) if critique else 0
                    }
                    for step_id, critique in critiques.items() if critique
                }
            }

        return results

    def generate_metrics(
        self,
        consensus_steps: List[Step]
    ) -> Dict[str, Any]:

        if not self.causal_attribution:
            raise ValueError("No analysis has been performed yet. Call analyze_trace() first.")

        metrics = {}

        # 1. Causal Attribution Metrics
        initial_identified_step_ids = self.causal_attribution.get_causal_steps()
        initial_identified_steps = [self.trace.get_step(step_id) for step_id in initial_identified_step_ids]
        causal_metrics = {
            "num_identified_causal_steps": len(initial_identified_steps),
            "identified_steps": [step.to_dict() for step in initial_identified_steps if step],
        }

        
        gt_set = set[int]([step.step_id for step in consensus_steps])
        id_set = set[int](initial_identified_step_ids)

        true_positives = len(gt_set & id_set)
        false_positives = len(id_set - gt_set)
        false_negatives = len(gt_set - id_set)

        precision = true_positives / len(id_set) if len(id_set) > 0 else 0.0
        recall = true_positives / len(gt_set) if len(gt_set) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        causal_metrics.update({
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "num_ground_truth_causal_steps": len(consensus_steps),
            "ground_truth_steps": [step.to_dict() for step in consensus_steps],
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        })

        metrics["causal_attribution_metrics"] = causal_metrics

        # 2. Repair Metrics
        repair_metrics = {
            "total_repairs_attempted": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "success_rate": 0.0,
            "repairs_by_step": {}
        }

        if self.counterfactual_repair:
            best_repairs = self.counterfactual_repair.get_all_best_repairs()
            all_repairs = self.counterfactual_repair.repairs

            total = sum(len(repair_list) for repair_list in all_repairs.values())
            successful = sum(
                1 for repair_list in all_repairs.values()
                for repair in repair_list
                if repair.success_predicted
            )

            repair_metrics.update({
                "total_repairs_attempted": total,
                "successful_repairs": successful,
                "failed_repairs": total - successful,
                "success_rate": round(successful / total, 4) if total > 0 else 0.0,
                "repairs_by_step": {
                    step_id: {
                        "success": repair.success_predicted,
                        "minimality_score": round(repair.minimality_score, 4)
                    }
                    for step_id, repair in best_repairs.items()
                }
            })

        metrics["repair_metrics"] = repair_metrics

        # 3. Minimality Metrics
        minimality_metrics = {
            "average_minimality": None,
            "min_minimality": None,
            "max_minimality": None,
            "minimality_by_step": {}
        }

        if self.counterfactual_repair:
            best_repairs = self.counterfactual_repair.get_all_best_repairs()
            if best_repairs:
                minimality_scores = [r.minimality_score for r in best_repairs.values()]
                minimality_metrics.update({
                    "average_minimality": round(sum(minimality_scores) / len(minimality_scores), 4),
                    "min_minimality": round(min(minimality_scores), 4),
                    "max_minimality": round(max(minimality_scores), 4),
                    "minimality_by_step": {
                        step_id: round(repair.minimality_score, 4)
                        for step_id, repair in best_repairs.items()
                    }
                })

        metrics["minimality_metrics"] = minimality_metrics

        # 4. Multi-Agent Agreement
        agreement_data = {
            "average_consensus_score": None,
            "num_steps_critiqued": 0,
            "steps_with_agreement": []
        }

        if self.multi_agent_critique:
            critique_results = self.multi_agent_critique.critique_results

            if critique_results:
                consensus_scores = [r.consensus_score for r in critique_results.values()]
                agreement_data["average_consensus_score"] = round(
                    sum(consensus_scores) / len(consensus_scores), 4
                )
                agreement_data["num_steps_critiqued"] = len(critique_results)

                for step_id, result in critique_results.items():
                    step_data = {
                        "step_id": step_id,
                        "consensus_score": round(result.consensus_score, 4),
                        "final_verdict": "CAUSAL" if result.final_verdict else "NOT CAUSAL",
                        "agent_a_score": self.causal_attribution.crs_scores.get(step_id, 0.0)
                    }

                    # Extract Agent B details
                    agent_b_critique = next(
                        (c for c in result.critiques if c["agent"] == "Agent_B"),
                        None
                    )
                    if agent_b_critique:
                        step_data["agent_b_agrees"] = agent_b_critique["agrees"]
                        step_data["agent_b_confidence"] = round(agent_b_critique["confidence"], 4)
                        step_data["agent_b_reasoning"] = self._extract_reasoning(
                            agent_b_critique["response"]
                        )

                    # Extract Agent C (final critic) details
                    agent_c_critique = next(
                        (c for c in result.critiques if c["agent"] == "Agent_C"),
                        None
                    )
                    if agent_c_critique:
                        step_data["agent_c_agrees"] = agent_c_critique["agrees"]
                        step_data["agent_c_confidence"] = round(agent_c_critique["confidence"], 4)
                        step_data["agent_c_reasoning"] = self._extract_reasoning(
                            agent_c_critique["response"]
                        )
                        step_data["final_critic_summary"] = self._extract_reasoning(
                            agent_c_critique["response"]
                        )

                    agreement_data["steps_with_agreement"].append(step_data)

        metrics["multi_agent_agreement"] = agreement_data

        return metrics

    def _extract_reasoning(self, response: str) -> str:
        if "REASONING:" in response:
            reasoning = response.split("REASONING:")[-1].strip()
            return reasoning
        return response