# Causal Skill Decomposition Report

- Experiment: MBPP
- Run ID: run_MBPP_2025-12-06T08:50:36.324654
- Total traces: 50
- Passing traces: 26
- Failing traces: 24
- Total steps analyzed: 300
- Causal steps labeled: 23
- Clusters (skills): 4
- Vocabulary size: 258
- Skill labeling model: google/gemini-3-flash-preview

## Skill 0
- Size: 8
- Causal rate: 1.0
- Repair success rate: 0.0
- Dominant step types: [('llm_response', 5), ('reasoning', 3)]
- Dominant tools: []
- Dominant skill labels: [('task decomposition', 2), ('algorithmic strategy selection', 1), ('structural requirement alignment', 1), ('edge case identification', 1), ('iterative algorithm implementation', 1)]
- Top terms: ['the', 'a', 'problem', 'to', 'of', 'sub', 'or', 'specific', 'goal', 'occurs']
- Examples:
  - [mbpp-601] step 2 llm_response  [algorithmic strategy selection]: ```python from typing import List, Tuple  def max_chain_length(pairs: List[Tuple[int, int]]) -> int:     pairs.sort(key=lambda x: x[1])     n = len(pairs)   ...
  - [mbpp-615] step 0 reasoning  [task decomposition]: Implement average_tuple for task mbpp-615. Run official tests to verify.
  - [mbpp-615] step 2 llm_response  [structural requirement alignment]: from typing import Tuple  def average_tuple(nums: Tuple[Tuple[float, ...], ...]) -> Tuple[float, ...]:     return tuple(sum(values) / len(values) for values ...

## Skill 1
- Size: 7
- Causal rate: 1.0
- Repair success rate: 0.0
- Dominant step types: [('llm_response', 7)]
- Dominant tools: []
- Dominant skill labels: [('code completion integrity', 2), ('algorithmic implementation accuracy', 1), ('structural schema inference', 1), ('semantic naming consistency', 1), ('code logic synthesis', 1)]
- Top terms: ['the', 'code', 'logic', 'requirements', 'type:llm_response', 'implementation', 'to', 'a', 'language', 'natural']
- Examples:
  - [mbpp-602] step 2 llm_response  [algorithmic implementation accuracy]: def first_repeated_char(s: str) -> str:     seen = set()     for char in s:         if char in seen:             return char         seen.add(char)     retur...
  - [mbpp-610] step 2 llm_response  [code completion integrity]: def remove_kth_element(lst, k):     """     Remove the k'th element from a given list.          Parameters:     lst (list): The list from which to remove the...
  - [mbpp-613] step 2 llm_response  [structural schema inference]: ```python def maximum_value(records):     """     Find the maximum value in record list as tuple attribute in the given tuple list.     Each element in recor...

## Skill 2
- Size: 3
- Causal rate: 1.0
- Repair success rate: 0.0
- Dominant step types: [('llm_response', 3)]
- Dominant tools: []
- Dominant skill labels: [('geometric constraint reasoning', 1), ('mathematical property derivation', 1), ('regex pattern completeness', 1)]
- Top terms: ['geometric', 'mathematical', 'the', 'for', 'property', 'nested', 'recursive', 'relationship', 'pattern', 'constraint']
- Examples:
  - [mbpp-626] step 2 llm_response  [geometric constraint reasoning]: import math  def triangle_area(r: float) -> float:     """     Find the largest triangle that can be inscribed in the semicircle of radius r.     The largest...
  - [mbpp-636] step 2 llm_response  [mathematical property derivation]: def Check_Solution(a, b, c):     """     Check if the roots of the quadratic equation ax^2 + bx + c = 0 are reciprocal of each other.     Returns True if the...
  - [mbpp-640] step 2 llm_response  [regex pattern completeness]: import re  def remove_parenthesis(s):     """     Remove content within parentheses including the parentheses themselves from the input string.     """     r...

## Skill 3
- Size: 5
- Causal rate: 1.0
- Repair success rate: 0.0
- Dominant step types: [('reasoning', 5)]
- Dominant tools: []
- Dominant skill labels: [('Task Decomposition', 3), ('task decomposition', 2)]
- Top terms: ['a', 'or', 'intermediate', 'single', 'execute', 'attempts', 'complex', 'defining', 'actionable', 'steps']
- Examples:
  - [mbpp-612] step 0 reasoning  [Task Decomposition]: Implement merge for task mbpp-612. Run official tests to verify.
  - [mbpp-613] step 0 reasoning  [Task Decomposition]: Implement maximum_value for task mbpp-613. Run official tests to verify.
  - [mbpp-620] step 0 reasoning  [task decomposition]: Implement largest_subset for task mbpp-620. Run official tests to verify.
