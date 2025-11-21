"""
MongoDB Storage Module for CausalFlow Traces and Failures

This module provides MongoDB integration for storing experiment runs with traces.

Collections:
- runs: Stores experiment runs with nested passing and failing traces
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import json
class MongoDBStorage:

    def __init__(self):
        load_dotenv()
        self.mongo_uri = os.getenv('MONGODB_URI')

        if not self.mongo_uri:
            raise ValueError(
                "MONGODB_URI not found. Please set it in .env file or pass it as parameter."
            )

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client.get_default_database()
        self.runs = self.db['runs']
        self._setup_indexes()
        self.current_run_id = None
    def _setup_indexes(self):
        self.runs.create_index([("run_id", ASCENDING)], unique=True)
        self.runs.create_index([("timestamp", DESCENDING)])
        self.runs.create_index([("experiment_name", ASCENDING)])

    def _convert_keys_to_strings(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._convert_keys_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_keys_to_strings(item) for item in obj]
        else:
            return obj

    def _parse_trace_json(self, trace_data: Any) -> Dict[str, Any]:

        if isinstance(trace_data, str):
            return json.loads(trace_data)
        return trace_data

    def create_run(
        self,
        experiment_name: str,
        num_problems: int
    ) -> str:

        timestamp = datetime.utcnow().isoformat()
        run_id = f"run_{experiment_name}_{timestamp}"

        document = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "num_problems": num_problems,
            "passing_traces": [],
            "failing_traces": [],
            "stats": {
                "total": 0,
                "passing": 0,
                "failing": 0
            }
        }

        self.runs.insert_one(document)
        self.current_run_id = run_id
        print(f"Created new run: {run_id}")
        return run_id

    def add_passing_trace(
        self,
        run_id: str,
        trace_data: Any,
        problem_id: Any,
        problem_statement: str,
        gold_answer: Any,
        final_answer: Any
    ):
        trace_obj = self._parse_trace_json(trace_data)

        trace_obj = self._convert_keys_to_strings(trace_obj)

        trace_document = {
            "problem_id": problem_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "problem_statement": problem_statement,
            "gold_answer": gold_answer,
            "final_answer": final_answer,
            "trace": trace_obj
        }

        self.runs.update_one(
            {"run_id": run_id},
            {
                "$push": {"passing_traces": trace_document},
                "$inc": {"stats.total": 1, "stats.passing": 1}
            }
        )

        print(f"Added passing trace for problem {problem_id} to run {run_id}")

    def add_failing_trace(
        self,
        run_id: str,
        trace_data: Any,
        problem_id: Any,
        problem_statement: str,
        gold_answer: Any,
        final_answer: Any,
        analysis_results: Dict[str, Any],
        metrics: Dict[str, Any]
    ):
        """
        Add a failing trace with complete CausalFlow analysis to an existing run.

        Args:
            run_id: Run identifier
            trace_data: Complete trace data (from TraceLogger.to_json())
            problem_id: Unique problem identifier
            problem_statement: The problem text
            gold_answer: Expected answer
            final_answer: Agent's answer (incorrect)
            analysis_results: Complete analysis results
            metrics: All metrics
        """
        trace_obj = self._parse_trace_json(trace_data)

        trace_obj = self._convert_keys_to_strings(trace_obj)
        analysis_results = self._convert_keys_to_strings(analysis_results)
        metrics = self._convert_keys_to_strings(metrics)

        trace_document = {
            "problem_id": problem_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "problem_statement": problem_statement,
            "gold_answer": gold_answer,
            "final_answer": final_answer,

            "trace": trace_obj,

            "analysis": {
                "causal_graph": analysis_results.get("causal_graph", {}),
                "causal_attribution": analysis_results.get("causal_attribution", {}),
                "counterfactual_repairs": analysis_results.get("counterfactual_repairs", {}),
                "multi_agent_critique": analysis_results.get("multi_agent_critique", {})
            },

            "metrics": {
                "minimality": {
                    "average": metrics['minimality_metrics'].get("average_minimality"),
                    "min": metrics['minimality_metrics'].get("min_minimality"),
                    "max": metrics['minimality_metrics'].get("max_minimality"),
                    "by_step": metrics['minimality_metrics'].get("minimality_by_step", {})
                },

                "attribution": {
                    "num_identified_causal_steps": metrics['causal_attribution_metrics'].get("num_identified_causal_steps"),
                    "identified_steps": metrics['causal_attribution_metrics'].get("identified_steps", []),
                    "precision": metrics['causal_attribution_metrics'].get("precision"),
                    "recall": metrics['causal_attribution_metrics'].get("recall"),
                    "f1_score": metrics['causal_attribution_metrics'].get("f1_score"),
                    "true_positives": metrics['causal_attribution_metrics'].get("true_positives"),
                    "false_positives": metrics['causal_attribution_metrics'].get("false_positives"),
                    "false_negatives": metrics['causal_attribution_metrics'].get("false_negatives"),
                    "num_ground_truth_causal_steps": metrics['causal_attribution_metrics'].get("num_ground_truth_causal_steps")
                },

                "repairs": {
                    "total_repairs_attempted": metrics['repair_metrics'].get("total_repairs_attempted"),
                    "successful_repairs": metrics['repair_metrics'].get("successful_repairs"),
                    "failed_repairs": metrics['repair_metrics'].get("failed_repairs"),
                    "success_rate": metrics['repair_metrics'].get("success_rate"),
                    "repairs_by_step": metrics['repair_metrics'].get("repairs_by_step", {})
                },

                "multi_agent": {
                    "average_consensus_score": metrics['multi_agent_agreement'].get("average_consensus_score"),
                    "num_steps_critiqued": metrics['multi_agent_agreement'].get("num_steps_critiqued"),
                    "steps_with_agreement": metrics['multi_agent_agreement'].get("steps_with_agreement", [])
                },

            },
        }

        self.runs.update_one(
            {"run_id": run_id},
            {
                "$push": {"failing_traces": trace_document},
                "$inc": {"stats.total": 1, "stats.failing": 1}
            }
        )

        print(f"Added failing trace with complete analysis for problem {problem_id} to run {run_id}")

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:

        return self.runs.find_one({"run_id": run_id})

    def get_run_statistics(self, run_id: str) -> Dict[str, Any]:

        run = self.get_run(run_id)
        if not run:
            return {}

        return {
            "run_id": run_id,
            "experiment_name": run.get("experiment_name"),
            "timestamp": run.get("timestamp"),
            "total_traces": run["stats"]["total"],
            "passing_traces": run["stats"]["passing"],
            "failing_traces": run["stats"]["failing"],
            "accuracy": run["stats"]["passing"] / run["stats"]["total"] if run["stats"]["total"] > 0 else 0
        }

    def close(self):
        self.client.close()
        print("MongoDB connection closed")
