import json
from typing import Any, Dict, List
from dataclasses import fields
from pathlib import Path
from evals.eval_envs.base_envs import Basic_Recorder

TASK_DESCRIPTION = {
    'dynamicmem': """The evaluation of downstream agent system is based on DynamicMem:
- Each task is a single user's digital life: months of app activity logs spanning apps like Fitbit, banking (Chase), calendar, messaging, and LLM assistants.
- The agent must answer natural-language questions about the user's habits, preferences, and behavioural patterns.
- Questions span life domains: work routines, health habits, financial behaviours, social activities, and leisure preferences.
- Different users have **entirely different lifestyles, occupations, and activity patterns** — memory must be personalised per user.
- The memory structure should strengthen the following abilities of the agent:
    - Personalised Profile Construction: building a structured user profile from raw app-log sequences (recorder.init['app_logs']).
    - Longitudinal Pattern Extraction: identifying recurring behaviours (weekly meetings, daily exercise, monthly financial transfers) from timestamped log entries.
    - Cross-Domain Synthesis: connecting behavioural patterns across different life domains (e.g. work stress → reduced fitness activity).
    - Temporal Precision: storing and retrieving when (day-of-week, time-of-day, frequency) specific behaviours occur.
    - Selective Retrieval: for a given QA question, surfacing only the relevant subset of the user profile rather than the entire log history.
""",
}

CHROMA_CHEETSHEET = """## Initialize Chroma DB

Use db = Chroma(embedding_function=Embedding()) to create the database. DO NOT use persist_dir.

### Add Memory: Adds new text entries to the database and returns their unique IDs.

db.add_texts(
    texts: List[str],
    metadatas: Optional[Union[str, int, float, bool, None]] = None,
    ids: Optional[List[str]] = None
) -> List[str]

- metadatas must be **flat list**: each value must be a single primitive type (str, int, float, bool, or None).
- You cannot pass lists, nested dicts, or other complex objects.
- If you need to store structured data, serialize it to a JSON string:

### Retrieve Memory

db.similarity_search(
    query: str,
    k: int = 4
) -> List[Document]

return List[Document]: [
  Document(
    page_content="the agent found a key",
    metadata={"type": "item"}
  )
]

### Get by ID
db.get(
    ids: Optional[List[str]] = None
) -> Dict[str, List]

### Delete Memory
db.delete(
    ids: Optional[List[str]] = None
) -> None
    """

NXGRAPH_CHEETSHEET = """
NETWORKX GRAPH CHEATSHEET

Context:
    import networkx as nx
    G = nx.Graph()

1. NODE OPERATIONS
- G.add_node(node, **attrs): Add a single node with optional attributes.
- G.add_nodes_from([n1, n2], **common_attrs): Add multiple nodes at once (shared attributes apply to all).
- G.remove_node(node): Remove a node and all edges connected to it.
- G.remove_nodes_from([n1, n2]): Remove multiple nodes.
- node in G: Check if a node exists.
- G.nodes: Get all nodes (NodeView).
- G.nodes[node]: Access node attributes as a dict.
- nx.set_node_attributes(G, {node: {"attr": value}}): Set attributes for nodes.

2. EDGE OPERATIONS
- G.add_edge(u, v, **attrs): Add an edge between two nodes.
- G.add_edges_from([(u, v), (x, y)], **attrs): Add multiple edges at once (shared attributes apply to all).
- G.remove_edge(u, v): Remove a single edge.
- G.remove_edges_from([(u, v), (x, y)]): Remove multiple edges.
- G.has_edge(u, v): Check if an edge exists.
- G.edges: Get all edges (EdgeView).
- G.edges[(u, v)]: Access edge attributes as a dict.
- nx.set_edge_attributes(G, {(u, v): {"weight": 1.0}}): Set attributes for edges.

3. TRAVERSAL / NEIGHBORHOOD
- G.neighbors(node): Get neighbors of a node.
- G.adj[node]: Get dict of neighbors with edge data.
- nx.shortest_path(G, source, target): Find one shortest path between nodes.
- nx.shortest_path_length(G, source, target): Get shortest path length.
- nx.all_simple_paths(G, source, target, cutoff): Generate all simple paths up to a cutoff length.
- nx.connected_components(G): Get connected components as node sets.
- G.subgraph([n1, n2, n3]): Extract a subgraph induced by given nodes.

4. ANALYSIS / CENTRALITY
- G.degree(node): Get degree (number of edges) for a single node.
- G.degree(): Get degree for all nodes (DegreeView).
- nx.degree_centrality(G): Compute degree centrality (dict of node -> score).
- nx.betweenness_centrality(G): Compute betweenness centrality.
- nx.pagerank(G): Compute PageRank scores for nodes.
- nx.clustering(G): Compute local clustering coefficient.
- nx.is_connected(G): Check if graph is connected.
- nx.number_connected_components(G): Count connected components.

5. UTILITIES
- G.copy(): Make a copy of the graph.
- G.clear(): Remove all nodes and edges.
- nx.to_dict_of_dicts(G): Convert graph to adjacency dict.
- nx.to_numpy_array(G): Get adjacency matrix as a NumPy array.
"""

TOOL_CHEETSHEET = """
TOOLS AVAILABLE:

1. Hire Agent
Class: Agent

- Purpose: Asynchronous wrapper around OpenAI Chat API, allows system/user prompts and optional JSON schema validation. Higher insights: this could be used to summarise information or gain new insights from them.
- Initialization:
    from utils.hire_agent import Agent
    agent = Agent(model: str, system_prompt: str, output_schema: Optional[Dict] = None)
- Key Methods:
    - agent.get_agent_config() -> Dict: Returns agent configuration and chat history.
    - await agent.ask(user_input: str, with_history: bool = False) -> Any:
        Sends a user message asynchronously and returns the model's response.
        If an output_schema is provided, response will be validated against it and returned as a JSON object.
- Usage Example:
    response = await agent.ask("Hello, how are you?")

-IMPORTANT:
    - output_schema must be a **json schema**. Format example:
{
    "location": {
        "type": "string",
        "description": "The location to get the weather for"
    },
    "unit": {
        "type": ["string", "null"],
        "description": "The unit to return the temperature in",
        "enum": ["F", "C"]
    }
}

    - model could be "gpt-5", "gpt-5-mini", based on whether you need reasoning ability.

2. Embedding
Class: Embedding

- Purpose: Async embedding manager for computing single or batch embeddings, with optional similarity calculation.
- Initialization:
    from utils.hire_agent import Embedding
    embedder = Embedding(model: str = "text-embedding-3-small", retries: int  = 3, retry_delay: float = 1.0)
- Key Methods:
    - await embedder.get_embedding(text: str) -> List[float]:
        Computes embedding for a single text string asynchronously.
    - await embedder.get_batch_embeddings(texts: List[str]) -> List[List[float]]:
        Computes embeddings for multiple texts asynchronously.
    - await Embedding.compute_similarity(emb1: List[float], emb2: List[float], metric: str = "cosine") -> float:
        Computes similarity between two embeddings asynchronously.
    - await Embedding.compute_one_to_group_similarity(emb: List[float], group_emb: List[List[float]], metric: str = "cosine") -> List[float]:
        Computes similarity between one embedding and a group of embeddings asynchronously.

- Notes:
    * All embedding calls automatically update a global token tracker.
    * Similarity functions support cosine similarity and run in parallel for efficiency.
"""


def build_analysis_prompt(memo_info):
    MEMO_ANALYSIS_OUTPUT_FORMAT = {
        "learned_from_suggestion_example": {
            "type": "string",
            "description": "Findings derived from the provided suggestion_example and improve_score. Bullet list of concrete factors (patterns) that made the suggestion succeed or fail. And principles to adopt when making future suggestions."
        },
        "trajectory_score_assessment": {
            "type": "array",
            "description": "Analyse each retrieved memory item based on the QA trajectories and benchmark scores.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["Useful", "Potentially Useful", "Irrelevant", "Empty/BadFormat"],
                        "description": "Categorization of the memory item's relevance: did it contain the facts needed to answer the question?"
                    },
                    "how_it_can_help": {
                        "type": "string",
                        "description": "If Useful/Potentially Useful: how it supports the answer. If Irrelevant/Empty: reason (wrong keying, over-specific, missing pattern, poor formatting)."
                    }
                },
                "required": ["label", "how_it_can_help"]
            }
        },
        "content_quality_issues": {
            "type": "string",
            "description": "Detected content-level problems (duplicates, empty entries, serialization issues, missing temporal precision). Why those harm retrieval or downstream QA.",
        },
        "structure_and_coherence": {
            "type": "string",
            "description": "Analysis of layer interactions, keying, and query-awareness. Which parts generalise across users, which are overfitted to a specific user's data.",
        },
        "suggested_changes": {
            "type": "array",
            "description": "Concrete changes to apply to the current memory structure code.",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "what": {"type": "string", "description": "Precise description of what to change."},
                    "why": {"type": "string", "description": "Link to observations/principles."}
                },
                "required": ["priority", "what", "why"]
            }
        }
    }

    example = memo_info.get('improve_example', {})
    example_block = f"""<Suggestion Example>
    Here is a previous suggestion, along with the code it looked at.
    The example includes a memory structure and its modification attempt, annotated with an improvement score (positive = improved, negative = degraded).
    Infer the underlying patterns that differentiate effective modifications from harmful ones, and apply this reasoning to suggest an improved modification for the current memory structure.
    {json.dumps(example, indent=1, ensure_ascii=False)}
    </Suggestion Example>
    """ if example else ''

    system_prompt = f"""You are a **Senior Agent Construction Engineer** responsible for providing suggestions for a memory structure written by an entry-level engineer, to make the memory structure better for a downstream QA agent to answer personalisation questions.

### Memo Information Overview
1. **source_code**
    - Each layer (inheriting `Sub_memo_layer`) contains:
      - **Retrieve**: fetches relevant memory elements from the database.
      - **Update**: writes or modifies entries in the database.
    - Final MemoStructure (inheriting `MemoStructure`) contains:
        - **general_update**: Phase 1 — called to build a user profile from app logs.
        - **general_retrieve**: Phase 2 — called once per QA question to retrieve relevant memory.
    - Two-phase protocol:
        - `general_update(recorder)`: called during Phase 1 with chunks of app logs.
          `recorder.init['app_logs']` = current batch of log entries; `recorder.init['user_profile']` = demographics.
          Expected behaviour: extract and store user habits, preferences, and behavioural patterns.
        - `general_retrieve(recorder)`: called during Phase 2 before each QA question.
          `recorder.init['query']` = current question; `recorder.init['app_logs']` = all logs (reference).
          Returns Dict passed to the answering agent.
          Expected behaviour: retrieve facts relevant to the query from the stored user profile.

2. **examples**
    - Each example is one sampled QA pair, containing:
        - `query`: the natural-language question asked
        - `retrieved_memory`: the dict returned by `general_retrieve` for this question
        - `predicted`: the agent's answer
        - `reference`: the ground-truth answer
        - `score`: integer 0–10, rated by an LLM judge with the following criteria:
            - 0–2: Mostly or completely incorrect. Contains major factual errors, contradicts the reference, or misses almost all key points.
            - 3–5: Partially correct but missing significant key points from the reference, or contains notable factual errors.
            - 6–8: Covers most key points from the reference correctly. Minor omissions or inaccuracies may exist, but core information is accurate.
            - 9–10: Fully covers all key points from the reference accurately. No factual errors or contradictions. (Extra details beyond the reference are acceptable.)
        - `judge_reason`: the LLM judge's brief explanation of why it gave that score — highlights which key points were hit or missed
        - `relevant_app_logs`: the specific app log entries that contain the ground-truth evidence for this question
    - Examples are sampled across score bins. Low-score examples are the primary diagnostic signal —
      compare `relevant_app_logs` against `retrieved_memory` to identify what the memory should have stored but didn't.
      Use `judge_reason` to understand exactly which key points were missing or incorrect.
    - High-score examples are provided as contrast — they show what the memory does well; avoid breaking these.

3. **benchmark_eval_score**
    - Mean QA score (0–10) across all users and questions. Use this to identify structural bottlenecks.

### Your Task:
Analyse past suggestion examples and current QA trajectories, then produce concrete, prioritised suggestions to improve the memory structure.

Step 1 — Learn from past suggestions
    1. Look at the provided improve_score and the suggestion_example that produced it.
    2. Explain why that suggestion led to improvement or degradation.
    3. Extract 2–5 general principles and 2–5 pitfalls.

Step 2 — Inspect QA trajectories and diagnose failures
    1. For each low-score example (score ≤ 5):
        - Read `judge_reason` to understand which key points the answer missed or got wrong.
        - Look at `relevant_app_logs`: these are the ground-truth log entries that contain the answer.
        - Look at `retrieved_memory`: this is what the memory actually returned.
        - Identify the gap: did the memory fail to store the right information in Phase 1, or fail to retrieve it in Phase 2?
    2. For each retrieved memory item, label it as Useful / Potentially Useful / Irrelevant / Empty/BadFormat.
    3. For high-score examples, note what the memory did well to avoid breaking it.

Step 3 — Inspect memory source and produce concrete suggestions
    1. Using Step 1 principles and Step 2 labels, propose specific code-level changes.
    2. For each suggestion: what to change, why it will help, priority (High/Medium/Low).
    3. Link to benchmark performance: which structural weaknesses correlate with low QA accuracy?

Extra checks:
    1. Flag obvious content issues: duplicates, empty entries, missing timestamps, mis-typed fields.
    2. Check layer interaction: do layers pass structured outputs, or dump free-form text?
    3. If retrieval returns empty dicts, prioritise structural fixes (keying, type consistency).

### Benchmark Information:
{TASK_DESCRIPTION['dynamicmem']}

### Required Output:
Return a JSON object following this schema:
{json.dumps(MEMO_ANALYSIS_OUTPUT_FORMAT, indent=2, ensure_ascii=False)}
"""

    user_prompt = f"""{example_block}
<CURRENT SOURCE CODE>
{memo_info["source_code"]}
</CURRENT SOURCE CODE>
<CURRENT QA TRAJECTORY EXAMPLES>
{json.dumps(memo_info["examples"], indent=2, ensure_ascii=False)}
</CURRENT QA TRAJECTORY EXAMPLES>
<CURRENT BENCHMARK SCORE>
{memo_info.get('benchmark_eval_score', {}).get('benchmark_overall_eval_score', 0.0)}
</CURRENT BENCHMARK SCORE>
"""
    return system_prompt, user_prompt, MEMO_ANALYSIS_OUTPUT_FORMAT


def get_metadata_dict(instance) -> dict:
    meta_dict = {}
    for f in fields(instance):
        meta_dict[f.name] = dict(f.metadata)
    return meta_dict


def build_generate_new_code_prompt(
    memo_info: Dict[str, Any],
    analysis_result: Dict[str, Any],
    recorder: Basic_Recorder,
):
    base_dir = Path(__file__).parent.parent / "evals" / "agents"
    file_path = base_dir / "memo_structure.py"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find memo_structure.py at {file_path.resolve()}")
    basic_classes = file_path.read_text(encoding="utf-8")

    interaction_recorder_info = get_metadata_dict(recorder)
    if analysis_result:
        suggestion = {
            'trajectory_score_assessment': analysis_result['trajectory_score_assessment'],
            'suggested_changes': analysis_result['suggested_changes'],
        }
    else:
        suggestion = {}

    interaction_prompt = f"""
    Your `general_retrieve` and `general_update` will take `Basic_Recorder` as input.
    The DynamicMemRecorder (a subclass of Basic_Recorder) has these attributes:
    {json.dumps(interaction_recorder_info, indent=2, ensure_ascii=False)}

    == DynamicMem Two-Phase Protocol ==

    Phase 1 — general_update(recorder):
      Called N times to build the user profile from app logs.
      recorder.init['app_logs']     = List[dict]  — current batch of app log entries
      recorder.init['user_profile'] = dict         — user demographic info (age, occupation, industry, etc.)
      Each app_log entry has: app_log_id, timestamp, app_name, api_name, request, response.
      Expected behaviour: extract and store user habits, preferences, and behavioural patterns into internal DB.
      NOTE: general_update may be called multiple times (once per chunk of logs); your internal DB must be
      persistent across calls (store it as self.* attributes on the MemoStructure instance).

    Phase 2 — general_retrieve(recorder):
      Called once per QA question (after Phase 1 is complete).
      recorder.init['query']        = str          — current natural-language question
      recorder.init['app_logs']     = List[dict]   — all app logs (reference only)
      recorder.init['user_profile'] = dict          — user demographic info
      Return value: Dict  — memory context passed directly to the answering agent.
      Expected behaviour: retrieve facts relevant to the query from the stored user profile.
      The returned dict is sent verbatim to the agent — keep it clean, structured, and non-redundant.
    """

    system_prompt = f"""You are a senior AI software engineer. Your task is to build an agent memory system composed of multiple specialised memory layers and a coordinating memory structure.
The agent will be used in the DynamicMem personalisation benchmark.
Your memory structure aims to help a downstream QA agent accurately answer questions about a specific user's habits, preferences, and behavioural patterns.

{TASK_DESCRIPTION['dynamicmem']}

You are given the following two base classes:
<BACKBONE_CODE>
{basic_classes}
</BACKBONE_CODE>

Inherit these base classes and import as follows:
```python
from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
from utils.hire_agent import Agent, Embedding
from langchain_chroma import Chroma
```

<CODE_INPUT>
{interaction_prompt}
</CODE_INPUT>

<CODE_USAGE>
Your memory structure will be used in the two-phase DynamicMem workflow:
    - `general_update(recorder)`: called during Phase 1 (once or multiple times) to ingest app logs
      and build a personalised user profile in the internal database.
    - `general_retrieve(recorder)`: called during Phase 2 (once per QA question) to retrieve
      relevant facts from the stored profile. The returned Dict is given directly to the answering agent.
    IMPORTANT: each user gets a fresh MemoStructure() instance — there is NO cross-user memory sharing.
    The same instance persists across all Phase 1 + Phase 2 calls for one user.
</CODE_USAGE>

Here are the basic tools provided:
<GRAPH_DATABASE_INTERACTION>
{NXGRAPH_CHEETSHEET}
</GRAPH_DATABASE_INTERACTION>

<CHROMA_DATABASE_INTERACTION>
{CHROMA_CHEETSHEET}
</CHROMA_DATABASE_INTERACTION>

<OTHER_TOOLS>
{TOOL_CHEETSHEET}
</OTHER_TOOLS>

### Your Task:
Modify or create code that fully satisfies the following design goals:

1. **Multiple Memory Layers:**
   - Create multiple subclasses of `Sub_memo_layer`.
   - Each layer must have a clear responsibility and maintain its own database (Chroma or NetworkX).
   - Think carefully about what type of behavioural data belongs in each layer
     (e.g., temporal patterns, cross-domain correlations, preference summaries).

2. **General Retrieve/Update Orchestration:**
   - Create a subclass of `MemoStructure` that orchestrates all layers.
   - `general_update()` should propagate app-log information to relevant layers.
   - `general_retrieve()` should chain layer outputs intelligently for the given query.
     Output from one layer can become input to the next.

3. **Personalisation and Temporal Precision:**
   - Store *when* behaviours occur (day-of-week, time-of-day, frequency), not just *what*.
   - Design retrieval to surface only the facts relevant to the current question.
   - Avoid returning the entire stored profile; be selective and query-aware.

4. **Out-of-the-Box Reasoning:**
   - Think about the semantic flow of information across layers.
   - Avoid hard-coded if/else branches for specific apps or domains.
   - Express generalizable principles: the structure should work for users with very different lifestyles.
   - Keep retrieved memory clean and useful — avoid repetition or truncated meaningful text.

5. **Integration with Utilities:**
   - Use Chroma for semantic similarity search (query-aware retrieval).
   - Use NetworkX for structural/relational patterns across apps or time slots.
   - Use Agent for LLM-based summarisation or pattern extraction if needed.

6. **Code Quality:**
   - Output clean, runnable Python code following PEP8.
   - `general_retrieve()` and `general_update()` accept a `Basic_Recorder` and orchestrate end-to-end.
   - Initialize all layers in `MemoStructure.__init__`.
   - Avoid placeholders like `pass` or `# TODO`.
   - Do not overuse defensive programming; raise exceptions for unexpected conditions.

The goal is to build a memory system that accurately captures individual user behaviour from app logs
and retrieves the right facts for any personalisation question.

### Important:
- Think creatively about data flow — outputs of one layer can feed into the next.
- Provide **only the final rewritten code**, no explanations.
"""

    if memo_info.get('source_code', ''):
        user_prompt = f"""
        Here is the current code that you must edit:
        <CURRENT_CODE>
        {memo_info.get('source_code', '')}
        </CURRENT_CODE>

        Here is the score of current code:
        <REWARD>
        {memo_info.get('benchmark_eval_score', {}).get('benchmark_overall_eval_score', 0.0)}
        </REWARD>

        Here is the analysis result (suggestions):
        <ANALYSIS_RESULT>
        {json.dumps(suggestion, ensure_ascii=False, indent=2)}
        </ANALYSIS_RESULT>
        """
    else:
        user_prompt = "Please generate new code based on your understanding of the task and requirements."

    return system_prompt, user_prompt


def build_reflection_prompt(code_str: str, recorder: Basic_Recorder, error_msg: str):
    base_dir = Path(__file__).parent.parent / "evals/agents"
    file_path = base_dir / "memo_structure.py"
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find memo_structure.py at {file_path.resolve()}")
    basic_classes = file_path.read_text(encoding="utf-8")

    interaction_recorder_info = get_metadata_dict(recorder)
    interaction_prompt = f"""
    Your `general_retrieve` and `general_update` will take `Basic_Recorder` as input:
    {json.dumps(interaction_recorder_info, indent=2, ensure_ascii=False)}

    == DynamicMem Two-Phase Protocol ==
    Phase 1 — general_update(recorder):
      recorder.init['app_logs']     = List[dict]  — current batch of app log entries
      recorder.init['user_profile'] = dict         — user demographics
    Phase 2 — general_retrieve(recorder):
      recorder.init['query']        = str          — current question
      recorder.init['app_logs']     = List[dict]   — all logs (reference)
      recorder.init['user_profile'] = dict          — user demographics
      Must return a Dict for the downstream agent.
    """

    system_prompt = f"""You are a senior AI software engineer and code repair expert.
Your role is to carefully analyse the provided code and error information, identify potential errors or design flaws, and directly rewrite or edit the code to fix those issues — while keeping the main design goals and intentions exactly the same.

You are given the following context and base classes:
<BACKBONE_CODE>
{basic_classes}
</BACKBONE_CODE>

<CODE_INPUT>
{interaction_prompt}
</CODE_INPUT>

<CODE_USAGE>
Your memory structure will be used in the DynamicMem two-phase workflow:
    - `general_update(recorder)`: Phase 1 — ingest app logs and build user profile.
    - `general_retrieve(recorder)`: Phase 2 — retrieve facts relevant to the query; return a Dict.
    IMPORTANT: each user gets a fresh MemoStructure() instance — no cross-user sharing.
</CODE_USAGE>

### Your Task:
Carefully inspect the code and error information, detect the root cause of errors, structural issues, or missing implementations, and **only fix the root cause code**.

<GRAPH_DATABASE_INTERACTION>
{NXGRAPH_CHEETSHEET}
</GRAPH_DATABASE_INTERACTION>

<CHROMA_DATABASE_INTERACTION>
{CHROMA_CHEETSHEET}
</CHROMA_DATABASE_INTERACTION>

<OTHER_TOOLS>
{TOOL_CHEETSHEET}
</OTHER_TOOLS>

### Output:
Return **only the final corrected Python code**, no explanations or commentary.
"""

    user_prompt = f"""
    Here's the code with potential error:
    <CODE_FOR_MODIFY>
    ```python
    {code_str}
    ```
    </CODE_FOR_MODIFY>
    And here's the corresponding error message:
    {error_msg}
    Find the root cause first and then modify only the corresponding code to avoid the error.
    """
    return system_prompt, user_prompt
