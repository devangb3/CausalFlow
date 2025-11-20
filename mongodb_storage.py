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
    """
    MongoDB storage manager for CausalFlow experiment runs and traces.

    Ensures:
    - No data is stripped or sliced
    - Complete storage of all reports and metrics
    - Traces stored as objects, not strings
    - Integer keys converted to strings for MongoDB compatibility
    """

    def __init__(self, mongo_uri: Optional[str] = None):
        """
        Initialize MongoDB connection.

        Args:
            mongo_uri: MongoDB connection URI. If None, loads from .env
        """
        # Load environment variables
        load_dotenv()

        # Get MongoDB URI
        self.mongo_uri = mongo_uri or os.getenv('MONGODB_URI')

        if not self.mongo_uri:
            raise ValueError(
                "MONGODB_URI not found. Please set it in .env file or pass it as parameter."
            )

        # Connect to MongoDB
        try:
            self.client = MongoClient(self.mongo_uri)
            # Test connection
            self.client.admin.command('ping')
            print(f"Successfully connected to MongoDB")
        except ConnectionFailure as e:
            raise ConnectionFailure(f"Failed to connect to MongoDB: {e}")

        # Get database (extract from URI or use default)
        self.db = self.client.get_default_database()

        # Setup collections
        self.runs = self.db['runs']

        # Create indexes for efficient querying
        self._setup_indexes()

        # Current run document
        self.current_run_id = None

    def _setup_indexes(self):
        """Create indexes for collections."""
        # Index on run_id (unique) and timestamp
        self.runs.create_index([("run_id", ASCENDING)], unique=True)
        self.runs.create_index([("timestamp", DESCENDING)])
        self.runs.create_index([("experiment_name", ASCENDING)])

        print("MongoDB indexes created successfully")

    def _convert_keys_to_strings(self, obj: Any) -> Any:
        """
        Recursively convert all dictionary keys to strings for MongoDB compatibility.
        MongoDB requires all keys to be strings, not integers.

        Args:
            obj: Object to convert

        Returns:
            Object with all keys converted to strings
        """
        if isinstance(obj, dict):
            return {str(k): self._convert_keys_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_keys_to_strings(item) for item in obj]
        else:
            return obj

    def _parse_trace_json(self, trace_data: Any) -> Dict[str, Any]:
        """
        Parse trace data if it's a JSON string, otherwise return as is.

        Args:
            trace_data: Trace data (could be string or dict)

        Returns:
            Parsed trace as dictionary
        """
        if isinstance(trace_data, str):
            return json.loads(trace_data)
        return trace_data

    def create_run(
        self,
        experiment_name: str,
        num_problems: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new experiment run document.

        Args:
            experiment_name: Name of the experiment (e.g., "GSM8K")
            num_problems: Total number of problems in this run
            metadata: Optional metadata for the run

        Returns:
            Run ID
        """
        timestamp = datetime.utcnow().isoformat()
        run_id = f"run_{experiment_name}_{timestamp}"

        document = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "num_problems": num_problems,
            "passing_traces": [],
            "failing_traces": [],
            "metadata": metadata or {},
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
        final_answer: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a passing trace to an existing run.

        Args:
            run_id: Run identifier
            trace_data: Complete trace data (from TraceLogger.to_json())
            problem_id: Unique problem identifier
            problem_statement: The problem text
            gold_answer: Expected answer
            final_answer: Agent's answer
            metadata: Optional additional metadata
        """
        # Parse trace if it's a JSON string
        trace_obj = self._parse_trace_json(trace_data)

        # Convert integer keys to strings
        trace_obj = self._convert_keys_to_strings(trace_obj)

        trace_document = {
            "problem_id": problem_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "problem_statement": problem_statement,
            "gold_answer": gold_answer,
            "final_answer": final_answer,
            "trace": trace_obj,  # Complete trace as object - never stripped
            "metadata": metadata or {}
        }

        # Add to run's passing_traces array and update stats
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
        metrics: Dict[str, Any],
        reports: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
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
            reports: All generated reports
            metadata: Optional additional metadata
        """
        # Parse trace if it's a JSON string
        trace_obj = self._parse_trace_json(trace_data)

        # Convert all integer keys to strings for MongoDB compatibility
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

            # Complete trace as object - never stripped or sliced
            "trace": trace_obj,

            # Complete analysis results - all components
            "analysis": {
                "causal_graph": analysis_results.get("causal_graph", {}),
                "causal_attribution": analysis_results.get("causal_attribution", {}),
                "counterfactual_repairs": analysis_results.get("counterfactual_repairs", {}),
                "multi_agent_critique": analysis_results.get("multi_agent_critique", {})
            },

            # All metrics - complete, never subset
            "metrics": {
                # Minimality metrics
                "minimality": {
                    "average": metrics.get("average_minimality"),
                    "min": metrics.get("min_minimality"),
                    "max": metrics.get("max_minimality"),
                    "by_step": metrics.get("minimality_by_step", {})
                },

                # Causal attribution metrics
                "attribution": {
                    "num_identified_causal_steps": metrics.get("num_identified_causal_steps"),
                    "identified_steps": metrics.get("identified_steps", []),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1_score": metrics.get("f1_score"),
                    "true_positives": metrics.get("true_positives"),
                    "false_positives": metrics.get("false_positives"),
                    "false_negatives": metrics.get("false_negatives"),
                    "num_ground_truth_causal_steps": metrics.get("num_ground_truth_causal_steps")
                },

                # Repair metrics
                "repairs": {
                    "total_repairs_attempted": metrics.get("total_repairs_attempted"),
                    "successful_repairs": metrics.get("successful_repairs"),
                    "failed_repairs": metrics.get("failed_repairs"),
                    "success_rate": metrics.get("success_rate"),
                    "repairs_by_step": metrics.get("repairs_by_step", {})
                },

                # Multi-agent agreement metrics
                "multi_agent": {
                    "average_consensus_score": metrics.get("average_consensus_score"),
                    "num_steps_critiqued": metrics.get("num_steps_critiqued"),
                    "steps_with_agreement": metrics.get("steps_with_agreement", [])
                },

                # Store complete original metrics object to ensure nothing is lost
                "complete_metrics": metrics
            },

            # All reports - full text, never stripped
            "reports": {
                "full_report": reports.get("full_report", ""),
                "attribution_report": reports.get("attribution_report", ""),
                "repair_report": reports.get("repair_report", ""),
                "critique_report": reports.get("critique_report", "")
            },

            "metadata": metadata or {}
        }

        # Add to run's failing_traces array and update stats
        self.runs.update_one(
            {"run_id": run_id},
            {
                "$push": {"failing_traces": trace_document},
                "$inc": {"stats.total": 1, "stats.failing": 1}
            }
        )

        print(f"Added failing trace with complete analysis for problem {problem_id} to run {run_id}")

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a complete run document.

        Args:
            run_id: Run identifier

        Returns:
            Run document or None if not found
        """
        return self.runs.find_one({"run_id": run_id})

    def get_run_statistics(self, run_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with statistics
        """
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

    def get_all_runs_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all runs.

        Returns:
            List of statistics for each run
        """
        runs = self.runs.find({}, {"run_id": 1, "experiment_name": 1, "timestamp": 1, "stats": 1})
        stats = []
        for run in runs:
            total = run["stats"]["total"]
            stats.append({
                "run_id": run["run_id"],
                "experiment_name": run.get("experiment_name"),
                "timestamp": run.get("timestamp"),
                "total_traces": total,
                "passing_traces": run["stats"]["passing"],
                "failing_traces": run["stats"]["failing"],
                "accuracy": run["stats"]["passing"] / total if total > 0 else 0
            })
        return stats

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics across all runs.

        Returns:
            Dictionary with aggregate statistics
        """
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_runs": {"$sum": 1},
                    "total_traces": {"$sum": "$stats.total"},
                    "total_passing": {"$sum": "$stats.passing"},
                    "total_failing": {"$sum": "$stats.failing"}
                }
            }
        ]

        result = list(self.runs.aggregate(pipeline))

        if not result:
            return {
                "total_runs": 0,
                "total_traces": 0,
                "total_passing_traces": 0,
                "total_failing_traces": 0
            }

        stats = result[0]
        return {
            "total_runs": stats["total_runs"],
            "total_traces": stats["total_traces"],
            "total_passing_traces": stats["total_passing"],
            "total_failing_traces": stats["total_failing"]
        }

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        print("MongoDB connection closed")
