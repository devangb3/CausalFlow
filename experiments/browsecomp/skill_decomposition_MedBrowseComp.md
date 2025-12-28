# Causal Skill Decomposition Report

- Experiment: MedBrowseComp
- Run ID: run_MedBrowseComp_2025-12-19T21:55:32.749095
- Total traces: 405
- Passing traces: 135
- Failing traces: 270
- Total steps analyzed: 3541
- Causal steps labeled: 167
- Skill groups: 6
- Skill labeling model: google/gemini-3-flash-preview
- Skill grouping model: google/gemini-3-flash-preview

## Source Relevance & Authority Assessment
- Description: The ability to critically evaluate search results to select the most authoritative, direct, and relevant source before committing to a deep fetch.
- Size: 25
- Repair success rate: 0.88
- Dominant step types: [('tool_call', 25)]
- Dominant tools: [('web_fetch', 24), ('web_search', 1)]
- Member skill labels:
  - Source selection strategy (count=5): The ability to evaluate search results and select the most authoritative or direct source for specific data points. Failure occurs when an agent commits to a single secondary source without verifying if it contains the specific metadata required.
  - premature source selection (count=5): The ability to critically evaluate search results for relevance and specificity before committing to a single source. Failure occurs when an agent selects the first available result without verifying if it meets all criteria of the complex query.
  - source relevance assessment (count=4): The ability to evaluate whether a specific source or URL likely contains the precise information required by the prompt before committing to a deep fetch. Failure occurs when an agent selects a source that does not match the specific comparative criteria or entities defined in the query.
  - Source verification and selection (count=3): The ability to evaluate search results and select the most authoritative or direct source for a specific data point. Failure occurs when an agent selects a secondary or generic landing page rather than a specific data-rich document.
  - source relevance evaluation (count=3): The ability to critically assess whether a specific resource or URL contains the precise comparative data required by the prompt before committing to a deep fetch. Failure occurs when an agent selects a source based on keyword overlap rather than verifying it addresses the specific experimental comparison requested.
  - source relevance verification (count=2): The ability to critically evaluate whether a specific source or URL contains the precise information required by the prompt before committing to a deep fetch. Failure occurs when an agent selects a source based on keyword overlap without verifying it matches the specific study or population criteria.
  - source relevance validation (count=1): The ability to critically evaluate search results to select the most specific and accurate source before initiating a deep fetch. Failure occurs when an agent selects a suboptimal or tangentially related resource based on keyword matching rather than precise criteria.
  - Comparative Evidence Evaluation (count=1): The ability to verify if a retrieved source satisfies all constraints of a multi-entity comparison. Failure occurs when an agent prematurely focuses on a source that only addresses one side of a comparative query.
  - Source Verification Strategy (count=1): The ability to select the most authoritative or direct source for specific metadata (like authorship) after identifying a candidate reference. Failure occurs when an agent redundantly fetches a known URL instead of extracting information from the existing search results or a more direct repository.
- Examples:
  - [medbrowsecomp_9] step 2 tool_call web_fetch [Source selection strategy]: 
  - [medbrowsecomp_61] step 2 tool_call web_fetch [source relevance validation]: 
  - [medbrowsecomp_95] step 4 tool_call web_fetch [Source verification and selection]: 

## Iterative Query Refinement & Narrowing
- Description: The ability to adjust search strategies based on initial results, maintaining a balance between search breadth and specific entity targeting.
- Size: 18
- Repair success rate: 0.7222
- Dominant step types: [('tool_call', 18)]
- Dominant tools: [('web_search', 18)]
- Member skill labels:
  - Search query refinement (count=7): The ability to accurately translate extracted information into specific search parameters. The failure occurs when an agent prematurely narrows a search to a specific entity (like a trial ID) based on incomplete or potentially incorrect prior evidence.
  - Search Query Diversification (count=3): The ability to reformulate search terms or pivot strategies when initial queries yield redundant results or technical barriers like access denials. Failure occurs when the agent repeats similar keyword strings instead of seeking alternative sources or metadata.
  - premature query narrowing (count=2): The ability to maintain a broad search strategy or verify multiple candidates before committing to a specific entity. Failure occurs when an agent fixates on a single search result as the definitive answer without sufficient evidence.
  - query parameter refinement (count=1): The ability to iteratively adjust search parameters based on initial results to narrow down specific data points. Failure occurs when the agent introduces unverified assumptions or specific constraints into a search query that were not present in the source material or previous results.
  - query parameter synthesis (count=1): The ability to accurately translate extracted evidence into search parameters. Failure occurs when the agent introduces incorrect entities or acronyms not present in the source text or task requirements.
  - query refinement specificity (count=1): The ability to narrow search parameters based on initial results to target a specific comparative study. Failure occurs when the agent shifts to a generic or tangentially related query instead of isolating the specific comparative variables requested.
  - premature hypothesis commitment (count=1): The ability to maintain an open search space before sufficient evidence is gathered. Failure occurs when an agent narrows its search to a specific entity or trial name without verifying it is the correct match for the query constraints.
  - premature query refinement (count=1): The ability to maintain search breadth before confirming specific entities. The failure occurs when an agent assumes a specific entity (like a trial name) is the correct target without verifying it first, leading to narrow and potentially irrelevant search results.
  - query refinement strategy (count=1): The ability to iteratively adjust search parameters based on previous negative results to narrow down specific identifiers. Failure occurs when the agent repeats similar search patterns or focuses on incorrect trial names without verifying the core comparison criteria.
- Examples:
  - [medbrowsecomp_11] step 4 tool_call web_search [Search query refinement]: 
  - [medbrowsecomp_27] step 2 tool_call web_search [query parameter refinement]: 
  - [medbrowsecomp_134] step 4 tool_call web_search [query parameter synthesis]: 

## Search Query Optimization & Formulation
- Description: The ability to translate complex user requirements into precise, effective search queries while avoiding broad or literal interpretations.
- Size: 18
- Repair success rate: 0.7222
- Dominant step types: [('tool_call', 18)]
- Dominant tools: [('web_search', 18)]
- Member skill labels:
  - Search query optimization (count=14): The ability to formulate precise search queries that include specific data requirements (like 'start date') to filter for high-relevance results. Failure occurs when a query is too broad, omitting key constraints from the prompt.
  - query formulation (count=1): The ability to translate a complex information need into a search query that captures all necessary constraints. Failure occurs when the query is too broad or lacks specific keywords required to isolate a unique target entity.
  - Comparative Query Formulation (count=1): The ability to translate specific comparative criteria into search terms that accurately reflect the relationship between entities. Failure occurs when the agent introduces incorrect variables or substitutes specific entities that were not present in the original requirements.
  - query optimization (count=1): The ability to refine search parameters to target specific identifiers or metadata when broad topical searches fail. The failure involves repeating long, verbatim strings from previous results rather than isolating key entities or publication markers.
  - literal query formulation (count=1): The ability to distinguish between a placeholder or negative condition and a literal search term. Failure occurs when the agent treats a logical negation or absence of a condition as a specific keyword, leading to irrelevant search results.
- Examples:
  - [medbrowsecomp_9] step 0 tool_call web_search [Search query optimization]: 
  - [medbrowsecomp_9] step 4 tool_call web_search [Search query optimization]: 
  - [medbrowsecomp_21] step 0 tool_call web_search [Search query optimization]: 

## Information Extraction & Parsing
- Description: The ability to accurately identify, isolate, and extract specific data points or identifiers from retrieved text or search results.
- Size: 11
- Repair success rate: 0.6364
- Dominant step types: [('tool_call', 9), ('reasoning', 2)]
- Dominant tools: [('web_fetch', 5), ('web_search', 4)]
- Member skill labels:
  - information extraction strategy (count=4): The ability to extract specific data points from already retrieved content rather than redundantly invoking external tools. This failure occurs when an agent performs a new search for information that is likely contained within the document it just fetched.
  - Targeted Information Retrieval (count=2): The ability to identify and access specific data sources or URLs that contain the precise information required by the prompt. Failure occurs when the agent selects a relevant document but fails to extract or navigate to the specific data point requested.
  - Entity Resolution (count=1): The ability to correctly map a specific query description to a unique identifier or resource. Failure occurs when the agent selects a specific URL or ID without sufficient evidence that it matches the target entity described in the prompt.
  - URL parameter synthesis (count=1): The ability to construct or predict valid resource identifiers based on search results. Failure occurs when an agent hallucinates a specific URL structure or document ID that does not exist or was not explicitly provided by a search tool.
  - Information Extraction (count=1): The ability to identify and isolate specific data points from a retrieved document or context. Failure occurs when the agent identifies the correct source but fails to accurately parse the required attribute from the text.
  - Entity attribute extraction (count=1): The ability to accurately identify and isolate specific metadata or sub-components (such as specific names in a list) from a retrieved text source. Failure occurs when the agent identifies the correct source but fails to correctly parse the specific detail requested.
  - URL parameter extraction (count=1): The ability to accurately extract and transfer specific identifiers from search results into tool arguments. Failure occurs when the agent identifies a relevant resource but fails to correctly map its unique identifier to the subsequent tool call.
- Examples:
  - [medbrowsecomp_35] step 2 tool_call web_fetch [Entity Resolution]: 
  - [medbrowsecomp_60] step 4 tool_call web_fetch [Targeted Information Retrieval]: 
  - [medbrowsecomp_75] step 2 tool_call web_fetch [URL parameter synthesis]: 

## Redundancy & Execution Efficiency
- Description: The ability to track state and context to avoid repeating tool calls or searching for information already present in the history.
- Size: 9
- Repair success rate: 0.6667
- Dominant step types: [('tool_call', 9)]
- Dominant tools: [('web_search', 6), ('web_fetch', 3)]
- Member skill labels:
  - redundant tool execution (count=4): The ability to recognize that a resource has already been accessed and its content is available, avoiding unnecessary repeated calls to the same endpoint.
  - information retrieval strategy (count=2): The ability to formulate search queries that target specific metadata or structural components of a document. Failure occurs when the agent uses a broad search for a specific data point already likely contained within its current context or accessible via more direct means.
  - redundant query generation (count=1): The ability to adapt search strategies when a specific source is inaccessible. Failure occurs when the agent repeats a search for information it already possesses or fails to pivot to alternative sources after a technical block.
  - redundant action avoidance (count=1): The ability to track execution history and avoid repeating identical tool calls that have already yielded insufficient or duplicate results. Failure occurs when the agent enters a loop of fetching the same resource without modifying parameters or strategy.
  - redundant query formulation (count=1): The ability to recognize when sufficient information has already been retrieved and to avoid generating search queries for data points already present in the current context.
- Examples:
  - [medbrowsecomp_39] step 4 tool_call web_fetch [redundant tool execution]: 
  - [medbrowsecomp_39] step 6 tool_call web_fetch [redundant tool execution]: 
  - [medbrowsecomp_162] step 8 tool_call web_search [redundant tool execution]: 

## Identifier & Resource Verification
- Description: The ability to confirm that a specific identifier, URL, or resource matches the target entity before proceeding with execution.
- Size: 2
- Repair success rate: 1.0
- Dominant step types: [('tool_call', 2)]
- Dominant tools: [('web_fetch', 2)]
- Member skill labels:
  - identifier verification (count=1): The ability to verify that a specific identifier or reference (such as a trial ID) matches the target criteria before proceeding with deep retrieval. The failure occurs when an agent pursues a specific resource without confirming it corresponds to the required entities or conditions.
  - Targeted URL Identification (count=1): The ability to extract and navigate to specific resource identifiers from search results to access primary data. Failure occurs when the agent correctly identifies a relevant source but must execute a direct fetch to retrieve granular details not present in the snippet.
- Examples:
  - [medbrowsecomp_10] step 8 tool_call web_fetch [identifier verification]: 
  - [medbrowsecomp_29] step 2 tool_call web_fetch [Targeted URL Identification]: 
