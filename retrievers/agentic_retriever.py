# retrievers/agentic_retriever.py

import re
import json
from core import Chunk, Retriever, Embedder, Index
from generation.local_generator import LocalLLMGenerator
from indexes.bm25_index import BM25Index


class AgenticRetriever(Retriever):
    """
    ReAct-style agentic retriever. LLM plans retrieval strategy per query.

    Available tools:
    - vector_search(query)        : semantic similarity search
    - keyword_search(query)       : BM25 sparse search
    - filter_search(query, type)  : vector search filtered by content_type
    - FINISH                      : stop and return accumulated results

    ReAct loop:
    1. LLM receives query + available tools
    2. LLM outputs: Thought + Action
    3. Action is executed → Observation (retrieved chunks)
    4. LLM sees Observation, outputs next Thought + Action
    5. Repeat until FINISH or max_steps

    Parameters
    ----------
    embedder     : Embedder
    dense_index  : Index     (FAISS or compatible)
    sparse_index : BM25Index (optional — if None, keyword_search unavailable)
    generator    : LocalLLMGenerator
    max_steps    : int       max ReAct iterations (default 4)
    fetch_k      : int       chunks per tool call
    """

    NAME = "agentic"

    SYSTEM_PROMPT = """You are a retrieval planning agent. Given a question, \
plan and execute searches to gather relevant information.

Available tools:
- vector_search(query): semantic similarity search — good for conceptual questions
- keyword_search(query): keyword/BM25 search — good for exact terms and entities
- filter_search(query, type): vector search filtered by type ('text' or 'table')
- FINISH: stop searching and return results

Format your response EXACTLY as:
Thought: <your reasoning>
Action: <tool_name>(<parameters>)

Or to finish:
Thought: <your reasoning>
Action: FINISH

Start immediately with 'Thought:'"""

    USER_TEMPLATE = """Question: {query}

{history}

What should I do next?"""

    def __init__(
        self,
        embedder:     Embedder,
        dense_index:  Index,
        sparse_index: BM25Index | None = None,
        generator:    LocalLLMGenerator | None = None,
        max_steps:    int = 4,
        fetch_k:      int = 3,
    ):
        self.embedder     = embedder
        self.dense_index  = dense_index
        self.sparse_index = sparse_index
        self.generator    = generator or LocalLLMGenerator(max_tokens=150)
        self.max_steps    = max_steps
        self.fetch_k      = fetch_k

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        all_chunks: dict[str, Chunk] = {}
        history    = ""
        step       = 0

        while step < self.max_steps:
            # LLM plans next action
            prompt    = self.USER_TEMPLATE.format(
                query   = query,
                history = history if history else "(No actions taken yet)",
            )
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
            response    = self.generator.call_raw(full_prompt)

            if not response:
                break

            # Parse Thought + Action
            thought, action, action_args = self._parse_response(response)

            if not action:
                break

            # FINISH signal
            if action.upper() == "FINISH":
                history += f"\nThought: {thought}\nAction: FINISH"
                break

            # Execute the tool
            retrieved, obs_text = self._execute_tool(
                action, action_args, query, tenant_id
            )

            # Accumulate results
            for chunk in retrieved:
                chunk.metadata["agentic_step"]   = step + 1
                chunk.metadata["agentic_action"] = action
                all_chunks[chunk.chunk_id]        = chunk

            # Update history for next iteration
            history += (
                f"\nThought: {thought}"
                f"\nAction: {action}({action_args})"
                f"\nObservation: {obs_text}"
            )
            step += 1

        # Return top_k by retrieval score
        ranked = sorted(
            all_chunks.values(),
            key=lambda c: c.metadata.get("score", 0),
            reverse=True,
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        action:      str,
        action_args: str,
        query:       str,
        tenant_id:   str,
    ) -> tuple[list[Chunk], str]:
        """Execute a tool call and return (chunks, observation_text)."""
        action = action.lower().strip()

        if action == "vector_search":
            search_query = action_args.strip().strip('"').strip("'") or query
            qvec    = self.embedder.embed_query(search_query)
            results = self.dense_index.search(
                qvec, top_k=self.fetch_k, tenant_id=tenant_id
            )
            obs = f"Found {len(results)} chunks. " + \
                  "; ".join(r.content[:60].strip() for r in results[:2])
            return results, obs

        elif action == "keyword_search":
            if not self.sparse_index:
                return [], "keyword_search unavailable — no BM25 index"
            search_query = action_args.strip().strip('"').strip("'") or query
            results = self.sparse_index.search_text(
                search_query, top_k=self.fetch_k, tenant_id=tenant_id
            )
            obs = f"Found {len(results)} chunks. " + \
                  "; ".join(r.content[:60].strip() for r in results[:2])
            return results, obs

        elif action == "filter_search":
            parts        = [p.strip().strip('"').strip("'")
                            for p in action_args.split(",")]
            search_query = parts[0] if parts else query
            filter_type  = parts[1] if len(parts) > 1 else "text"
            qvec    = self.embedder.embed_query(search_query)
            results = self.dense_index.search(
                qvec, top_k=self.fetch_k * 3, tenant_id=tenant_id
            )
            filtered = [r for r in results
                        if r.metadata.get("content_type") == filter_type][:self.fetch_k]
            obs = f"Found {len(filtered)} {filter_type} chunks."
            return filtered, obs

        else:
            return [], f"Unknown tool: {action}"

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str, str]:
        """Extract (thought, action_name, action_args) from ReAct response."""
        thought = ""
        action  = ""
        args    = ""

        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)",
                                  response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.+)", response, re.IGNORECASE)
        if action_match:
            action_str = action_match.group(1).strip()

            if action_str.upper().startswith("FINISH"):
                return thought, "FINISH", ""

            # Parse tool_name(args)
            fn_match = re.match(r"(\w+)\(([^)]*)\)", action_str)
            if fn_match:
                action = fn_match.group(1)
                args   = fn_match.group(2)
            else:
                action = action_str.split("(")[0].strip()
                args   = query   # fallback

        return thought, action, args