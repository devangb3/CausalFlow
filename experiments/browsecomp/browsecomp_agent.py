import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from llm_client import LLMClient
from trace_logger import TraceLogger, Step, StepType
from .web_env import WebEnvironment, SearchResponse, FetchResult
from .browsecomp_eval import grade_response

@dataclass
class BrowseCompExecutionContext:
    problem_id: str
    question: str
    gold_answer: str
    max_steps: int = 15


class BrowseCompAgent:
    
    SYSTEM_PROMPT = """You are a web browsing agent tasked with answering questions by searching the web and examining pages.

You have access to the following tools:
1. search(query) - Search the web for information
2. open_url(url) - Fetch and read the content of a specific URL
3. extract - Note important facts from the current context
4. answer - Provide your final answer when you have enough information

For each step, you must decide on ONE action to take. Think carefully about what information you need and how to find it efficiently.

Important guidelines:
- Start by searching for relevant information
- Open URLs that seem most likely to contain the answer
- Extract key facts as you find them
- When you have enough information, provide your answer with confidence
- Be thorough but efficient - don't waste steps on irrelevant searches
- If you cannot find the answer after several attempts, provide your best guess with low confidence"""

    def __init__(
        self,
        llm_client: LLMClient,
        web_env: WebEnvironment,
        max_steps: int = 15,
    ):
        self.llm = llm_client
        self.web_env = web_env
        self.max_steps = max_steps
        self._current_context: Optional[BrowseCompExecutionContext] = None
    
    def solve(
        self,
        problem_id: str,
        question: str,
        gold_answer: str,
    ) -> TraceLogger:
        context = BrowseCompExecutionContext(
            problem_id=problem_id,
            question=question,
            gold_answer=gold_answer,
            max_steps=self.max_steps,
        )
        self._current_context = context
        
        trace = TraceLogger(
            problem_statement=question,
            gold_answer=gold_answer,
        )
        
        return self._run(trace, context)
    
    def run_remaining_steps(self, history: List[Step]) -> TraceLogger:
        if not history:
            raise ValueError("History cannot be empty for run_remaining_steps")
        
        context = self._ensure_context()
        trace = self._build_trace_with_history(history, context)
        
        last_step = history[-1]
        
        if last_step.step_type == StepType.FINAL_ANSWER:
            return trace
        
        gathered_facts, search_history, page_summaries = self._reconstruct_state_from_history(history)
        
        if last_step.step_type == StepType.TOOL_CALL:
            try:
                gathered_facts, search_history, page_summaries = self._execute_pending_tool_call(
                    trace=trace,
                    tool_call_step=last_step,
                    gathered_facts=gathered_facts,
                    search_history=search_history,
                    page_summaries=page_summaries,
                )
            except Exception as e:
                trace.log_reasoning(
                    f"Error executing tool call: {e}. Forcing answer.",
                    dependencies=[last_step.step_id],
                )
                new_trace = self._force_answer(trace, context, gathered_facts, search_history=search_history, page_summaries=page_summaries, dependencies=[last_step.step_id])
                new_trace.success = grade_response(context.gold_answer, new_trace.final_answer, self.llm)
                return new_trace

        new_trace = self._run_with_state(
            trace=trace,
            context=context,
            gathered_facts=gathered_facts,
            search_history=search_history,
            page_summaries=page_summaries,
        )
        new_trace.success = grade_response(context.gold_answer, new_trace.final_answer, self.llm)
        return new_trace

    def _execute_pending_tool_call(
        self,
        trace: TraceLogger,
        tool_call_step: Step,
        gathered_facts: List[str],
        search_history: List[str],
        page_summaries: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:

        tool_name = tool_call_step.tool_name
        tool_args = tool_call_step.tool_args or {}
        
        if tool_name == "web_search":
            query = tool_args.get("query", "")
            try:
                search_result = self.web_env.web_search(query)
                result_summary = self._format_search_results(search_result)
                tool_success = True
            except Exception as e:
                result_summary = f"Search failed: {e}"
                tool_success = False
            
            tool_response_step = trace.log_tool_response(
                tool_name="web_search",
                dependencies=[tool_call_step.step_id],
                tool_call_result=tool_success,
                tool_output=result_summary,
            )
            
            search_history.append(f"Searched: {query}")
            if tool_success:
                page_summaries.append(f"Search results for '{query}':\n{result_summary}")
            
            self._attach_state_snapshot(
                trace, tool_response_step,
                gathered_facts, search_history, page_summaries,
            )
            
        elif tool_name == "web_fetch":
            url = tool_args.get("url", "")
            try:
                fetch_result = self.web_env.web_fetch(url)
                if fetch_result.error:
                    result_summary = f"Fetch error: {fetch_result.error}"
                    tool_success = False
                else:
                    result_summary = self._format_fetch_result(fetch_result)
                    tool_success = True
            except Exception as e:
                result_summary = f"Fetch failed: {e}"
                tool_success = False
            
            tool_response_step = trace.log_tool_response(
                tool_name="web_fetch",
                dependencies=[tool_call_step.step_id],
                tool_call_result=tool_success,
                tool_output=result_summary,
            )
            
            if tool_success:
                page_summaries.append(f"Page content from {url}:\n{result_summary}")
            
            self._attach_state_snapshot(
                trace, tool_response_step,
                gathered_facts, search_history, page_summaries,
            )
        else:
            raise ValueError(f"Unknown tool for execution: {tool_name}")
        
        return gathered_facts, search_history, page_summaries
    
    def _ensure_context(self) -> BrowseCompExecutionContext:
        if self._current_context is None:
            raise RuntimeError(
                "No execution context available. Run solve(...) before branching."
            )
        return self._current_context
    
    def _build_trace_with_history(
        self,
        history: List[Step],
        context: BrowseCompExecutionContext,
    ) -> TraceLogger:
        trace = TraceLogger(
            problem_statement=context.question,
            gold_answer=context.gold_answer,
        )
        
        if not history:
            return trace
        
        validated_history = self._validate_history(history)
        trace.steps = validated_history
        trace.current_step_id = len(validated_history)
        
        return trace
    
    def _validate_history(self, history: List[Step]) -> List[Step]:
        cloned_history = deepcopy(history)
        for idx, step in enumerate(cloned_history):
            if step.step_id != idx:
                raise ValueError(
                    f"History step_id sequence is non-contiguous at index {idx}: "
                    f"found {step.step_id}."
                )
            if step.step_type == StepType.FINAL_ANSWER:
                raise ValueError(
                    "History already contains a FINAL_ANSWER step; cannot resume."
                )
        return cloned_history
    
    def _attach_state_snapshot(
        self,
        trace: TraceLogger,
        step_id: int,
        gathered_facts: List[str],
        search_history: List[str],
        page_summaries: List[str],
    ) -> None:
        step = trace.get_step(step_id)
        if step is None:
            raise ValueError(f"Step {step_id} not found in trace")
        step.state_snapshot = {
            "gathered_facts": list(gathered_facts),
            "search_history": list(search_history),
            "page_summaries": list(page_summaries),
        }
    
    def _reconstruct_state_from_history(
        self,
        history: List[Step],
    ) -> Tuple[List[str], List[str], List[str]]:
        if not history:
            return [], [], []
        
        last_step = history[-1]
        if last_step.state_snapshot is None:
            return [], [], []
        
        snapshot = last_step.state_snapshot
        return (
            list(snapshot["gathered_facts"]),
            list(snapshot["search_history"]),
            list(snapshot["page_summaries"]),
        )
    
    def _run(
        self,
        trace: TraceLogger,
        context: BrowseCompExecutionContext,
    ) -> TraceLogger:
        return self._run_with_state(
            trace=trace,
            context=context,
            gathered_facts=[],
            search_history=[],
            page_summaries=[],
        )
    
    def _run_with_state(
        self,
        trace: TraceLogger,
        context: BrowseCompExecutionContext,
        gathered_facts: List[str],
        search_history: List[str],
        page_summaries: List[str],
    ) -> TraceLogger:

        dependencies = [trace.steps[-1].step_id] if trace.steps else []
        step_count = len(trace.steps)
        
        while step_count < context.max_steps:
            prompt = self._build_step_prompt(
                context=context,
                gathered_facts=gathered_facts,
                search_history=search_history,
                page_summaries=page_summaries,
                steps_remaining=context.max_steps - step_count,
            )
            try:
                agent_step = self.llm.generate_structured(
                    prompt,
                    schema_name="browsecomp_step",
                    system_message=self.SYSTEM_PROMPT,
                    temperature=0.0,
                )
            except Exception as e:
                trace.log_reasoning(
                    f"Error getting next action: {e}. Forcing answer.",
                    dependencies=dependencies,
                )
                return self._force_answer(trace, context, gathered_facts, search_history=search_history, page_summaries=page_summaries, dependencies=dependencies)
            
            action_type = agent_step.action_type
            
            if action_type == "search":
                query = agent_step.query
                if not query:
                    continue
                
                tool_call_step = trace.log_tool_call(
                    tool_name="web_search",
                    tool_args={"query": query},
                    dependencies=dependencies,
                )
                
                self._attach_state_snapshot(
                    trace, tool_call_step,
                    gathered_facts, search_history, page_summaries,
                )
                
                try:
                    search_result = self.web_env.web_search(query)
                    result_summary = self._format_search_results(search_result)
                    tool_success = True
                except Exception as e:
                    result_summary = f"Search failed: {e}"
                    tool_success = False
                
                tool_response_step = trace.log_tool_response(
                    tool_name="web_search",
                    dependencies=[tool_call_step],
                    tool_call_result=tool_success,
                    tool_output=result_summary,
                )
                
                search_history.append(f"Searched: {query}")
                if tool_success:
                    page_summaries.append(f"Search results for '{query}':\n{result_summary}")
                
                self._attach_state_snapshot(
                    trace, tool_response_step,
                    gathered_facts, search_history, page_summaries,
                )
                
                dependencies = [tool_response_step]
                
            elif action_type == "open_url":
                url = agent_step.url
                if not url:
                    raise ValueError("open_url action requires a url")
                
                tool_call_step = trace.log_tool_call(
                    tool_name="web_fetch",
                    tool_args={"url": url},
                    dependencies=dependencies,
                )
                
                self._attach_state_snapshot(
                    trace, tool_call_step,
                    gathered_facts, search_history, page_summaries,
                )
                
                try:
                    fetch_result = self.web_env.web_fetch(url)
                    if fetch_result.error:
                        result_summary = f"Fetch error: {fetch_result.error}"
                        tool_success = False
                    else:
                        result_summary = self._format_fetch_result(fetch_result)
                        tool_success = True
                except Exception as e:
                    result_summary = f"Fetch failed: {e}"
                    tool_success = False
                
                tool_response_step = trace.log_tool_response(
                    tool_name="web_fetch",
                    dependencies=[tool_call_step],
                    tool_call_result=tool_success,
                    tool_output=result_summary,
                )
                
                if tool_success:
                    page_summaries.append(f"Page content from {url}:\n{result_summary}")
                
                self._attach_state_snapshot(
                    trace, tool_response_step,
                    gathered_facts, search_history, page_summaries,
                )
                
                dependencies = [tool_response_step]
                
            elif action_type == "extract":
                facts = agent_step.extracted_facts or []
                note = agent_step.note
                
                reasoning_step = trace.log_reasoning(
                    f"Extracting facts: {note}\nFacts: {facts}",
                    dependencies=dependencies,
                )
                
                gathered_facts.extend(facts)
                
                self._attach_state_snapshot(
                    trace, reasoning_step,
                    gathered_facts, search_history, page_summaries,
                )
                
                dependencies = [reasoning_step]
                
            elif action_type == "answer":
                exact_answer = agent_step.exact_answer
                if not exact_answer:
                    raise ValueError("answer action requires exact_answer")
                
                explanation = agent_step.explanation or agent_step.note
                confidence = agent_step.confidence or 50.0
                
                formatted_response = self._format_final_response(
                    explanation=explanation,
                    exact_answer=exact_answer,
                    confidence=confidence,
                )
                
                final_step_id = trace.log_final_answer(formatted_response, dependencies=dependencies)
                
                self._attach_state_snapshot(
                    trace, final_step_id,
                    gathered_facts, search_history, page_summaries,
                )
                
                trace.record_outcome(
                    final_answer=exact_answer,
                    gold_answer=context.gold_answer,
                )
                
                return trace
            
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            step_count = len(trace.steps)
        
        return self._force_answer(trace, context, gathered_facts, search_history=search_history, page_summaries=page_summaries, dependencies=dependencies)
    
    def _force_answer(
        self,
        trace: TraceLogger,
        context: BrowseCompExecutionContext,
        gathered_facts: List[str],
        search_history: List[str],
        page_summaries: List[str],
        dependencies: List[int],
    ) -> TraceLogger:
        prompt = f"""You have reached the maximum number of steps to solve the question. Based on the information gathered, 
provide your best answer to the question.

Question: {context.question}

Facts gathered:
{chr(10).join(f'- {fact}' for fact in gathered_facts) if gathered_facts else '- No facts gathered'}

Search history:
{chr(10).join(f'- {s}' for s in search_history) if search_history else '- No searches performed'}

Page summaries:
{chr(10).join(f'- {s}' for s in page_summaries) if page_summaries else '- No page summaries'}

You MUST provide an answer now, even if uncertain. Use action_type: "answer" with your best guess."""
        
        try:
            agent_step = self.llm.generate_structured(
                prompt,
                schema_name="browsecomp_step",
                system_message=self.SYSTEM_PROMPT,
                temperature=0.0,
            )
            
            exact_answer = agent_step.exact_answer or "Unable to determine"
            explanation = agent_step.explanation or agent_step.note or "Forced answer due to step limit"
            confidence = agent_step.confidence or 10.0
            
        except Exception:
            exact_answer = "Unable to determine"
            explanation = "Failed to generate answer"
            confidence = 0.0
        
        formatted_response = self._format_final_response(
            explanation=explanation,
            exact_answer=exact_answer,
            confidence=confidence,
        )
        
        final_step_id = trace.log_final_answer(formatted_response, dependencies=dependencies)
        
        self._attach_state_snapshot(
            trace, final_step_id,
            gathered_facts, search_history, page_summaries,
        )
        
        trace.record_outcome(
            final_answer=exact_answer,
            gold_answer=context.gold_answer,
        )
        
        return trace
    
    def _build_step_prompt(
        self,
        context: BrowseCompExecutionContext,
        gathered_facts: List[str],
        search_history: List[str],
        page_summaries: List[str],
        steps_remaining: int,
    ) -> str:
        prompt_parts = [
            f"Question to answer: {context.question}",
            "",
            f"Steps remaining: {steps_remaining}",
            "",
        ]
        
        if search_history:
            prompt_parts.extend([
                "Previous searches:",
                *[f"  - {s}" for s in search_history[-5:]],  # Last 5 searches
                "",
            ])
        
        if gathered_facts:
            prompt_parts.extend([
                "Facts gathered so far:",
                *[f"  - {f}" for f in gathered_facts[-10:]],  # Last 10 facts
                "",
            ])
        
        if page_summaries:
            prompt_parts.extend([
                "Recent page content (truncated):",
                page_summaries[-1][:3000] if page_summaries else "",  # Most recent page
                "",
            ])
        
        prompt_parts.extend([
            "Decide on your next action. Choose ONE of:",
            "  - search: Search the web for information",
            "  - open_url: Fetch a specific URL to read its content",
            "  - extract: Note important facts from the current context",
            "  - answer: Provide your final answer (when you have enough information)",
            "",
            "Respond with your action in the required JSON format.",
        ])
        
        if steps_remaining <= 3:
            prompt_parts.append(
                "\nWARNING: Running low on steps. Consider providing your answer soon."
            )
        
        return "\n".join(prompt_parts)
    
    def _format_search_results(self, search_response: SearchResponse) -> str:
        lines = [f"Search results for: {search_response.query}"]
        for result in search_response.results[:10]:
            lines.append(f"\n{result.rank}. {result.title}")
            lines.append(f"   URL: {result.url}")
            lines.append(f"   {result.snippet}")
        return "\n".join(lines)
    
    def _format_fetch_result(self, fetch_result: FetchResult) -> str:
        lines = [
            f"Page: {fetch_result.title or fetch_result.url}",
            f"URL: {fetch_result.final_url}",
            f"Status: {fetch_result.status_code}",
            "",
            "Content:",
            fetch_result.text_content,
        ]
        return "\n".join(lines)
    
    def _format_final_response(
        self,
        explanation: str,
        exact_answer: str,
        confidence: float,
    ) -> str:
        return f"""Explanation: {explanation}
Exact Answer: {exact_answer}
Confidence: {confidence:.0f}%"""
    
    def get_execution_context_dict(self) -> Dict[str, Any]:
        if self._current_context is None:
            return {}
        
        return {
            "problem_id": self._current_context.problem_id,
            "question": self._current_context.question,
            "gold_answer": self._current_context.gold_answer,
            "max_steps": self._current_context.max_steps,
        }
