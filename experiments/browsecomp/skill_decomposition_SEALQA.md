# Causal Skill Decomposition Report

- Experiment: SealQA
- Run ID: run_SealQA_2025-12-18T02:41:05.742380
- Total traces: 240
- Passing traces: 100
- Failing traces: 140
- Total steps analyzed: 2698
- Causal steps labeled: 67
- Skill groups: 4
- Skill labeling model: google/gemini-3-flash-preview
- Skill grouping model: google/gemini-3-flash-preview

## Query Precision and Formulation
- Description: The ability to translate complex information needs into effective, specific, and well-structured search strings.
- Size: 23
- Repair success rate: 0.7391
- Dominant step types: [('tool_call', 23)]
- Dominant tools: [('web_search', 23)]
- Member skill labels:
  - search query optimization (count=12): The ability to formulate precise search queries that target specific data points required by a multi-part question. Failure occurs when a query is too broad or fails to include key constraints, such as the distinction between 'total sites' and 'cultural sites'.
  - information retrieval strategy (count=8): The ability to formulate effective search queries that target specific record-breaking entities or historical data. Failure occurs when a query is too broad or lacks the specificity needed to disambiguate between similar categories or historical milestones.
  - query formulation strategy (count=1): The ability to decompose a complex, multi-entity comparison into targeted search queries rather than inputting a long, natural language question that relies on the search engine to perform the intersection.
  - query formulation precision (count=1): The ability to translate complex natural language constraints into effective search queries. Failure occurs when the agent conflates distinct criteria into a single search string, reducing the specificity of results.
  - query formulation (count=1): The ability to translate a complex natural language information need into an effective search string. Failure occurs when the agent uses a literal or overly specific phrase that may limit search engine results compared to keyword-based or entity-focused queries.
- Examples:
  - [sealqa_0] step 0 tool_call web_search [information retrieval strategy]: 
  - [sealqa_15] step 0 tool_call web_search [search query optimization]: 
  - [sealqa_25] step 0 tool_call web_search [information retrieval strategy]: 

## Source Selection and Extraction Efficiency
- Description: The ability to identify authoritative sources and extract specific data points while avoiding redundant or high-volume, low-value actions.
- Size: 14
- Repair success rate: 0.6429
- Dominant step types: [('tool_call', 14)]
- Dominant tools: [('web_fetch', 9), ('web_search', 5)]
- Member skill labels:
  - source selection strategy (count=5): The ability to identify and navigate to the most granular or relevant data source based on initial search results. Failure occurs when an agent selects a broad overview page instead of a specialized sub-page (e.g., a list of episodes) that contains the specific structured data required.
  - redundant search execution (count=2): The ability to recognize when sufficient information has already been retrieved and avoid unnecessary tool calls. This failure occurs when an agent repeats a search for data already present in the current context or previous tool outputs.
  - unstructured data extraction (count=2): The ability to identify and retrieve specific data points from large, complex documents or PDFs. Failure occurs when an agent repeatedly attempts to fetch high-volume files without a strategy for parsing or locating the specific information within them.
  - source relevance evaluation (count=1): The ability to assess whether a specific source or URL is likely to contain factual information about a general knowledge query versus technical or meta-data. Failure occurs when an agent attempts to extract primary facts from non-authoritative or unrelated technical documents.
  - redundant information retrieval (count=1): The ability to recognize when sufficient information has already been gathered or when a new search query/URL will yield duplicate data. Failure occurs when the agent repeatedly accesses different sources for the same dataset instead of processing existing results.
  - information extraction strategy (count=1): The ability to parse large documents for specific data points and pivot to more granular search queries when initial broad searches fail to yield structured comparisons. The failure involves repeating broad queries instead of extracting specific entities for targeted verification.
  - redundant query generation (count=1): The ability to recognize when previous search results already contain the necessary information and to avoid generating repetitive queries that do not add new information.
  - redundant tool execution (count=1): The ability to recognize when information has already been retrieved or is currently being processed, avoiding unnecessary repeat queries to the same or similar tools.
- Examples:
  - [sealqa_25] step 2 tool_call web_fetch [source selection strategy]: 
  - [sealqa_25] step 5 tool_call web_search [redundant search execution]: 
  - [sealqa_69] step 6 tool_call web_fetch [source relevance evaluation]: 

## Iterative Search and Refinement
- Description: The ability to adapt search strategies based on previous results, including diversifying keywords and narrowing focus to fill specific data gaps.
- Size: 7
- Repair success rate: 0.8571
- Dominant step types: [('tool_call', 7)]
- Dominant tools: [('web_search', 6), ('web_fetch', 1)]
- Member skill labels:
  - Search query diversification (count=2): The ability to vary search parameters and keywords when initial attempts yield redundant or insufficient information. Failure occurs when the agent repeatedly uses similar queries that lead back to the same unhelpful source.
  - targeted search refinement (count=2): The ability to narrow down a broad search by identifying specific missing entities or dates from previous results to verify a sequence. Failure occurs when the agent prematurely focuses on a specific candidate without sufficient comparative evidence.
  - sequential information retrieval (count=1): The ability to systematically identify and access multiple distinct data sources required to satisfy a multi-part query. Failure occurs when an agent fails to iterate through all necessary sub-components of a set after identifying their existence.
  - Hypothesis-driven search refinement (count=1): The ability to formulate specific search queries based on inferred missing information or potential candidates when general queries fail to yield updated results. Failure occurs when the agent prematurely narrows the search to a specific unverified entity without sufficient evidence.
  - iterative query refinement (count=1): The ability to modify search parameters based on previous results to narrow down a specific chronological or ordinal fact. Failure occurs when the agent repeats similar queries without addressing the specific missing data point needed to complete a sequence.
- Examples:
  - [sealqa_25] step 7 tool_call web_fetch [sequential information retrieval]: 
  - [sealqa_106] step 10 tool_call web_search [Search query diversification]: 
  - [sealqa_149] step 4 tool_call web_search [Hypothesis-driven search refinement]: 

## Temporal and Contextual Grounding
- Description: The ability to resolve time-based constraints and verify premises before executing searches to ensure chronological accuracy.
- Size: 6
- Repair success rate: 0.8333
- Dominant step types: [('tool_call', 6)]
- Dominant tools: [('web_search', 5), ('web_fetch', 1)]
- Member skill labels:
  - temporal query refinement (count=2): The ability to adjust search parameters based on the recency of the target information and the failure of previous broad searches. This failure occurs when an agent prematurely narrows a search to a specific future or current season before establishing a baseline of recent historical data.
  - temporal grounding validation (count=1): The ability to verify the chronological relevance and existence of a resource before attempting access. This failure involves requesting data from a future date relative to the current real-world or simulation time.
  - temporal constraint grounding (count=1): The ability to translate relative time references (e.g., 'last year') into absolute values based on the current execution context. Failure occurs when the agent uses literal phrasing instead of resolving the specific date required for accurate retrieval.
  - temporal query optimization (count=1): The ability to formulate search queries that account for the most recent available data or specific timeframes. Failure occurs when an agent assumes a specific year for a 'most recent' event without verifying the current date or the actual latest occurrence of a recurring event.
  - premise verification (count=1): The ability to verify the accuracy of an assumed relationship or event before using it as a search constraint. The failure involves hallucinating a specific successor or cause for a record change without evidence.
- Examples:
  - [sealqa_5] step 4 tool_call web_fetch [temporal grounding validation]: 
  - [sealqa_16] step 0 tool_call web_search [temporal constraint grounding]: 
  - [sealqa_19] step 0 tool_call web_search [temporal query optimization]: 
