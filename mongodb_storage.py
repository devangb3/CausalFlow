"""
MongoDB Storage Module for CausalFlow Traces and Failures

This module provides MongoDB integration for storing:
- Passing traces
- Failing traces with complete analysis (reports, metrics, attributions, repairs)

Collections:
- passing_traces: Successful execution traces
- failing_traces: Failed traces with full CausalFlow analysis
"""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from dotenv import load_dotenv
import json


class MongoDBStorage:
    """
    MongoDB storage manager for CausalFlow traces and analysis results.

    Ensures:
    - No data is stripped or sliced
    - No duplicate storage
    - Complete storage of all reports and metrics
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
        self.passing_traces = self.db['passing_traces']
        self.failing_traces = self.db['failing_traces']

        # Create indexes for efficient querying and duplicate prevention
        self._setup_indexes()

    def _setup_indexes(self):
        """Create indexes for collections."""
        # Index on trace_id (unique) and timestamp for both collections
        self.passing_traces.create_index([("trace_id", ASCENDING)], unique=True)
        self.passing_traces.create_index([("timestamp", ASCENDING)])
        self.passing_traces.create_index([("problem_id", ASCENDING)])

        self.failing_traces.create_index([("trace_id", ASCENDING)], unique=True)
        self.failing_traces.create_index([("timestamp", ASCENDING)])
        self.failing_traces.create_index([("problem_id", ASCENDING)])

        print("MongoDB indexes created successfully")

    def _generate_trace_id(self, problem_id: Any, timestamp: str) -> str:
        """
        Generate a unique trace ID.

        Args:
            problem_id: Problem identifier
            timestamp: ISO format timestamp

        Returns:
            Unique trace ID
        """
        return f"trace_{problem_id}_{timestamp}"

    def save_passing_trace(
        self,
        trace_data: Dict[str, Any],
        problem_id: Any,
        problem_statement: str,
        gold_answer: Any,
        final_answer: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a passing trace to MongoDB.

        Args:
            trace_data: Complete trace data (from TraceLogger.to_json())
            problem_id: Unique problem identifier
            problem_statement: The problem text
            gold_answer: Expected answer
            final_answer: Agent's answer
            metadata: Optional additional metadata

        Returns:
            True if saved successfully, False if already exists
        """
        timestamp = datetime.utcnow().isoformat()
        trace_id = self._generate_trace_id(problem_id, timestamp)

        document = {
            "trace_id": trace_id,
            "problem_id": problem_id,
            "timestamp": timestamp,
            "success": True,
            "problem_statement": problem_statement,
            "gold_answer": gold_answer,
            "final_answer": final_answer,
            "trace": trace_data,  # Complete trace - never stripped
            "metadata": metadata or {}
        }

        try:
            self.passing_traces.insert_one(document)
            print(f"Saved passing trace: {trace_id}")
            return True
        except DuplicateKeyError:
            print(f"Trace {trace_id} already exists in passing_traces")
            return False

    def save_failing_trace(
        self,
        trace_data: Dict[str, Any],
        problem_id: Any,
        problem_statement: str,
        gold_answer: Any,
        final_answer: Any,
        analysis_results: Dict[str, Any],
        metrics: Dict[str, Any],
        reports: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a failing trace with complete CausalFlow analysis to MongoDB.

        Args:
            trace_data: Complete trace data (from TraceLogger.to_json())
            problem_id: Unique problem identifier
            problem_statement: The problem text
            gold_answer: Expected answer
            final_answer: Agent's answer (incorrect)
            analysis_results: Complete analysis results including:
                - causal_graph
                - causal_attribution
                - counterfactual_repairs
                - multi_agent_critique
            metrics: All metrics including:
                - minimality_scores (average, min, max, by_step)
                - causal_attribution_metrics (precision, recall, F1)
                - repair_metrics (success_rate, attempts)
                - multi_agent_agreement_metrics
            reports: All generated reports:
                - full_report: Complete text report
                - attribution_report: Causal attribution details
                - repair_report: Counterfactual repair details
                - critique_report: Multi-agent critique details
            metadata: Optional additional metadata

        Returns:
            True if saved successfully, False if already exists
        """
        timestamp = datetime.utcnow().isoformat()
        trace_id = self._generate_trace_id(problem_id, timestamp)

        document = {
            "trace_id": trace_id,
            "problem_id": problem_id,
            "timestamp": timestamp,
            "success": False,
            "problem_statement": problem_statement,
            "gold_answer": gold_answer,
            "final_answer": final_answer,

            # Complete trace - never stripped or sliced
            "trace": trace_data,

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
                    "by_step": metrics.get("minimality_by_step", [])
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
                    "repairs_by_step": metrics.get("repairs_by_step", [])
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

        try:
            self.failing_traces.insert_one(document)
            print(f"Saved failing trace with complete analysis: {trace_id}")
            return True
        except DuplicateKeyError:
            print(f"Trace {trace_id} already exists in failing_traces")
            return False

    def trace_exists(self, trace_id: str, collection: str = "both") -> bool:
        """
        Check if a trace exists in the database.

        Args:
            trace_id: Trace identifier
            collection: "passing", "failing", or "both"

        Returns:
            True if trace exists
        """
        if collection == "both":
            return (
                self.passing_traces.find_one({"trace_id": trace_id}) is not None or
                self.failing_traces.find_one({"trace_id": trace_id}) is not None
            )
        elif collection == "passing":
            return self.passing_traces.find_one({"trace_id": trace_id}) is not None
        elif collection == "failing":
            return self.failing_traces.find_one({"trace_id": trace_id}) is not None
        else:
            raise ValueError(f"Invalid collection: {collection}")

    def get_trace(self, trace_id: str, collection: str = "both") -> Optional[Dict[str, Any]]:
        """
        Retrieve a trace from the database.

        Args:
            trace_id: Trace identifier
            collection: "passing", "failing", or "both"

        Returns:
            Trace document or None if not found
        """
        if collection == "both":
            trace = self.passing_traces.find_one({"trace_id": trace_id})
            if trace:
                return trace
            return self.failing_traces.find_one({"trace_id": trace_id})
        elif collection == "passing":
            return self.passing_traces.find_one({"trace_id": trace_id})
        elif collection == "failing":
            return self.failing_traces.find_one({"trace_id": trace_id})
        else:
            raise ValueError(f"Invalid collection: {collection}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored traces.

        Returns:
            Dictionary with count statistics
        """
        return {
            "total_passing_traces": self.passing_traces.count_documents({}),
            "total_failing_traces": self.failing_traces.count_documents({}),
            "total_traces": (
                self.passing_traces.count_documents({}) +
                self.failing_traces.count_documents({})
            )
        }

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        print("MongoDB connection closed")
