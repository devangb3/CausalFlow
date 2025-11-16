"""
MultiAgentCritique: Validates causal attributions through multi-agent consensus.

This module implements the multi-agent critique component of CausalFlow
as specified in Section 8 of the research proposal.
"""

from typing import Dict, List, Any, Tuple, Optional
from trace_logger import TraceLogger, Step
from causal_attribution import CausalAttribution
from llm_client import MultiAgentLLM, LLMClient


class CritiqueResult:
    """Represents the result of a multi-agent critique."""

    def __init__(
        self,
        step_id: int,
        proposed_by: str,
        critiques: List[Dict[str, Any]],
        consensus_score: float,
        final_verdict: bool
    ):
        """
        Initialize critique result.

        Args:
            step_id: The step being critiqued
            proposed_by: Which agent proposed this as causal
            critiques: List of critique responses from agents
            consensus_score: Agreement score (0-1)
            final_verdict: Whether the step is confirmed as causal
        """
        self.step_id = step_id
        self.proposed_by = proposed_by
        self.critiques = critiques
        self.consensus_score = consensus_score
        self.final_verdict = final_verdict

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "proposed_by": self.proposed_by,
            "critiques": self.critiques,
            "consensus_score": self.consensus_score,
            "final_verdict": self.final_verdict
        }

    def __repr__(self) -> str:
        """String representation."""
        return (f"CritiqueResult(step_id={self.step_id}, "
                f"consensus={self.consensus_score:.2f}, "
                f"verdict={self.final_verdict})")


class MultiAgentCritique:
    """
    Employs multiple LLM agents to critique and validate causal attributions.

    Process:
    1. Agent A (proposer) identifies causal steps
    2. Agent B (critic 1) critiques the proposal
    3. Agent C (critic 2) provides additional critique
    4. System synthesizes consensus

    This triangulation reduces variance and improves robustness.
    """

    def __init__(
        self,
        trace: TraceLogger,
        causal_attribution: CausalAttribution,
        multi_agent_llm: Optional[MultiAgentLLM] = None,
        num_agents: int = 3
    ):
        """
        Initialize multi-agent critique system.

        Args:
            trace: The execution trace
            causal_attribution: Causal attribution results
            multi_agent_llm: Optional multi-agent LLM system
            num_agents: Number of agents to use (if creating new multi-agent system)
        """
        self.trace = trace
        self.causal_attribution = causal_attribution

        if multi_agent_llm is None:
            self.multi_agent_llm = MultiAgentLLM(num_agents=num_agents)
        else:
            self.multi_agent_llm = multi_agent_llm

        self.num_agents = self.multi_agent_llm.num_agents

        self.critique_results: Dict[int, CritiqueResult] = {}

    def critique_causal_attributions(
        self,
        step_ids: Optional[List[int]] = None
    ) -> Dict[int, CritiqueResult]:
        """
        Perform multi-agent critique on causal attributions.

        Args:
            step_ids: Optional list of step IDs to critique (if None, uses all causal steps)

        Returns:
            Dictionary mapping step_id to critique results
        """
        if step_ids is None:
            step_ids = self.causal_attribution.get_causal_steps()

        for step_id in step_ids:
            self.critique_results[step_id] = self._critique_step(step_id)

        return self.critique_results

    def _critique_step(self, step_id: int) -> CritiqueResult:
        """
        Perform multi-agent critique on a single step.

        Args:
            step_id: The step to critique

        Returns:
            Critique result
        """
        step = self.trace.get_step(step_id)
        if not step:
            return CritiqueResult(step_id, "unknown", [], 0.0, False)

        # Agent A (proposer) - already done by causal attribution
        crs_score = self.causal_attribution.crs_scores.get(step_id, 0.0)

        # Generate critiques from multiple agents
        critiques = []

        # Agent B (first critic)
        critique_b = self._generate_critique(step_id, agent_index=1, role="critic")
        critiques.append({
            "agent": "Agent_B",
            "role": "critic",
            "response": critique_b["response"],
            "agrees": critique_b["agrees"],
            "confidence": critique_b["confidence"]
        })

        # Agent C (second critic)
        critique_c = self._generate_critique(
            step_id,
            agent_index=2,
            role="meta-critic",
            previous_critique=critique_b["response"]
        )
        critiques.append({
            "agent": "Agent_C",
            "role": "meta-critic",
            "response": critique_c["response"],
            "agrees": critique_c["agrees"],
            "confidence": critique_c["confidence"]
        })

        # Synthesize consensus
        consensus_score = self._calculate_consensus(crs_score, critiques)
        final_verdict = consensus_score >= 0.5

        return CritiqueResult(
            step_id=step_id,
            proposed_by="Agent_A",
            critiques=critiques,
            consensus_score=consensus_score,
            final_verdict=final_verdict
        )

    def _generate_critique(
        self,
        step_id: int,
        agent_index: int,
        role: str,
        previous_critique: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a critique from a specific agent.

        Args:
            step_id: The step being critiqued
            agent_index: Index of the agent to use
            role: Role of this agent (critic, meta-critic)
            previous_critique: Optional previous critique to consider

        Returns:
            Dictionary with critique response, agreement, and confidence
        """
        step = self.trace.get_step(step_id)
        crs_score = self.causal_attribution.crs_scores.get(step_id, 0.0)

        prompt = self._create_critique_prompt(
            step, crs_score, role, previous_critique
        )

        try:
            agent = self.multi_agent_llm.get_agent(agent_index)
            response = agent.generate(
                prompt,
                system_message=f"You are a critical evaluator (role: {role}) analyzing causal claims.",
                temperature=0.3  # Lower temperature for more consistent critique
            )

            # Parse response for agreement and confidence
            agrees = self._parse_agreement(response)
            confidence = self._parse_confidence(response)

            return {
                "response": response,
                "agrees": agrees,
                "confidence": confidence
            }

        except Exception as e:
            print(f"Error generating critique for step {step_id}: {e}")
            return {
                "response": f"Error: {str(e)}",
                "agrees": False,
                "confidence": 0.0
            }

    def _create_critique_prompt(
        self,
        step: Step,
        crs_score: float,
        role: str,
        previous_critique: Optional[str] = None
    ) -> str:
        """
        Create a prompt for agent critique.

        Args:
            step: The step being critiqued
            crs_score: Causal Responsibility Score from Agent A
            role: Role of this critique
            previous_critique: Optional previous critique

        Returns:
            Prompt string
        """
        prompt = f"""You are critically evaluating a causal attribution claim.

EXECUTION TRACE SUMMARY:
- Task outcome: FAILED
- Final answer: {self.trace.final_answer}
- Correct answer: {self.trace.gold_answer}
- Total steps: {len(self.trace.steps)}

CAUSAL CLAIM:
Agent A claims that Step {step.step_id} is causally responsible for the failure.
Causal Responsibility Score (CRS): {crs_score}

STEP DETAILS:
- Step ID: {step.step_id}
- Type: {step.step_type.value}
- Content: {self._summarize_step(step)}
- Dependencies: {step.dependencies}

DESCENDANTS (affected by this step):
"""
        # Add descendant information
        descendants = list(self.causal_attribution.causal_graph.get_descendants(step.step_id))
        for desc_id in descendants[:5]:  # Limit to first 5
            desc_step = self.trace.get_step(desc_id)
            if desc_step:
                prompt += f"  Step {desc_id}: {self._summarize_step(desc_step)}\n"

        if previous_critique:
            prompt += f"\nPREVIOUS CRITIQUE:\n{previous_critique}\n"

        prompt += f"""
YOUR TASK ({role}):
Critically evaluate whether Step {step.step_id} is truly causally responsible for the failure.

Consider:
1. Is this step actually incorrect or problematic?
2. Would fixing this step alone lead to success?
3. Are there alternative explanations for the failure?
4. Is the causal chain from this step to the final failure clear?

Provide your critique in the following format:

AGREEMENT: [AGREE/DISAGREE/PARTIAL]
CONFIDENCE: [0.0-1.0]
REASONING: [Your detailed reasoning]

Be thorough and critical. Challenge weak causal claims.
"""

        return prompt

    def _summarize_step(self, step: Step) -> str:
        """
        Summarize a step for the prompt.

        Args:
            step: The step to summarize

        Returns:
            Summary string
        """
        return self.causal_attribution._summarize_step(step)

    def _parse_agreement(self, response: str) -> bool:
        """
        Parse whether the agent agrees with the causal claim.

        Args:
            response: The agent's response

        Returns:
            True if agrees, False otherwise
        """
        response_upper = response.upper()

        if "AGREEMENT: AGREE" in response_upper or "AGREEMENT:AGREE" in response_upper:
            return True
        elif "AGREEMENT: PARTIAL" in response_upper or "AGREEMENT:PARTIAL" in response_upper:
            return True  # Count partial as agreement
        else:
            return False

    def _parse_confidence(self, response: str) -> float:
        """
        Parse confidence score from response.

        Args:
            response: The agent's response

        Returns:
            Confidence score (0-1)
        """
        import re

        # Look for CONFIDENCE: X.X pattern
        match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)

        if match:
            try:
                confidence = float(match.group(1))
                return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except ValueError:
                pass

        # Default: medium confidence if no explicit score
        return 0.5

    def _calculate_consensus(
        self,
        crs_score: float,
        critiques: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate consensus score from multiple agents.

        Args:
            crs_score: Original CRS from Agent A
            critiques: List of critiques from other agents

        Returns:
            Consensus score (0-1)
        """
        # Weight each agent's input
        weights = {
            "Agent_A": 0.4,  # Proposer
            "Agent_B": 0.3,  # First critic
            "Agent_C": 0.3   # Second critic
        }

        # Agent A's score (proposer)
        total_score = weights["Agent_A"] * crs_score

        # Add weighted critique scores
        for critique in critiques:
            agent = critique["agent"]
            agrees = critique["agrees"]
            confidence = critique["confidence"]

            # Score is 1.0 if agrees, 0.0 if disagrees, weighted by confidence
            critique_score = (1.0 if agrees else 0.0) * confidence

            total_score += weights.get(agent, 0.1) * critique_score

        return max(0.0, min(1.0, total_score))

    def get_consensus_causal_steps(self, threshold: float = 0.5) -> List[int]:
        """
        Get steps confirmed as causal by consensus.

        Args:
            threshold: Minimum consensus score

        Returns:
            List of step IDs with consensus >= threshold
        """
        return [
            step_id for step_id, result in self.critique_results.items()
            if result.consensus_score >= threshold
        ]

    def generate_report(self) -> str:
        """
        Generate a human-readable report of multi-agent critique.

        Returns:
            Report string
        """
        lines = ["=" * 60]
        lines.append("MULTI-AGENT CRITIQUE REPORT")
        lines.append("=" * 60)
        lines.append(f"Number of Agents: {self.num_agents}")
        lines.append(f"Steps Critiqued: {len(self.critique_results)}")

        consensus_steps = self.get_consensus_causal_steps()
        lines.append(f"Steps Confirmed by Consensus: {len(consensus_steps)}")
        lines.append("")

        for step_id in sorted(self.critique_results.keys()):
            result = self.critique_results[step_id]
            step = self.trace.get_step(step_id)

            lines.append(f"\nStep {step_id} ({step.step_type.value}):")
            lines.append("-" * 60)
            lines.append(f"Content: {self._summarize_step(step)}")
            lines.append(f"Proposed by: {result.proposed_by}")
            lines.append(f"Consensus Score: {result.consensus_score:.2f}")
            lines.append(f"Final Verdict: {'CAUSAL' if result.final_verdict else 'NOT CAUSAL'}")
            lines.append("")

            for critique in result.critiques:
                lines.append(f"  {critique['agent']} ({critique['role']}):")
                lines.append(f"    Agrees: {critique['agrees']}")
                lines.append(f"    Confidence: {critique['confidence']:.2f}")
                reasoning = critique['response'].split('REASONING:')[-1].strip()[:150]
                lines.append(f"    Reasoning: {reasoning}...")
                lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (f"MultiAgentCritique(num_agents={self.num_agents}, "
                f"critiqued_steps={len(self.critique_results)})")
