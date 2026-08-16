# indexes/raptor_index.py

import uuid
import time
import numpy as np
from core import Chunk, Index, Embedder, Generator


class RAPTORIndex(Index):
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
    (Sarthi et al., Stanford 2024)

    Builds a multi-level summary tree over chunks:
    - Level 0: original chunks (leaves)
    - Level 1: LLM summaries of K-Means clusters of level-0 chunks
    - Level 2+: summaries of summaries (until one root remains)

    All nodes (leaves + summaries) are indexed in one flat FAISS index.
    Retrieval searches all levels simultaneously.

    Why this helps:
    - Specific queries match leaves (exact facts)
    - Thematic queries match summaries (high-level concepts)
    - Multi-hop queries match intermediate summaries

    Parameters
    ----------
    embedder         : Embedder  same embedder as pipeline
    generator        : Generator used to summarize clusters
    n_clusters       : int       clusters per level (default 3)
    max_levels       : int       max tree depth (default 3)
    min_cluster_size : int       don't cluster if fewer nodes than this
    """

    NAME = "raptor"

    SUMMARIZE_PROMPT = """Summarize the following text passages into a single coherent paragraph.
Preserve all key facts, concepts, and technical details.
The summary will be used for retrieval, so be specific and informative.

Passages:
{passages}

Summary:"""

    def __init__(
        self,
        embedder:         Embedder,
        generator:        Generator,
        n_clusters:       int = 3,
        max_levels:       int = 3,
        min_cluster_size: int = 2,
    ):
        import faiss
        self.embedder         = embedder
        self.generator        = generator
        self.n_clusters       = n_clusters
        self.max_levels       = max_levels
        self.min_cluster_size = min_cluster_size

        self._all_nodes: list[Chunk] = []     # all nodes across all levels
        self._faiss_index             = None   # built after add()
        self._dimension: int | None   = None

    def add(self, chunks: list[Chunk]) -> None:
        """Build the RAPTOR tree from leaf chunks."""
        if not chunks:
            return

        # Ensure leaves have embeddings
        leaves = [c for c in chunks if c.embedding]
        if not leaves:
            print("[RAPTOR] No embedded chunks — embed chunks before adding.")
            return

        self._dimension = len(leaves[0].embedding)

        # Tag leaves as level 0
        for leaf in leaves:
            leaf.metadata["raptor_level"] = 0
            leaf.metadata["raptor_node_type"] = "leaf"

        self._all_nodes = list(leaves)

        # Build tree recursively
        current_level_nodes = leaves
        for level in range(1, self.max_levels + 1):
            if len(current_level_nodes) < self.min_cluster_size:
                break

            print(f"  [RAPTOR] Building level {level} "
                  f"({len(current_level_nodes)} nodes → {self.n_clusters} clusters)...")

            summary_nodes = self._build_level(current_level_nodes, level)
            if not summary_nodes:
                break

            self._all_nodes.extend(summary_nodes)
            current_level_nodes = summary_nodes

        # Index all nodes (leaves + all summary levels) in FAISS
        self._build_faiss_index()

        print(f"  [RAPTOR] Tree complete: {len(leaves)} leaves + "
              f"{len(self._all_nodes) - len(leaves)} summary nodes "
              f"= {len(self._all_nodes)} total nodes indexed")

    def search(
        self,
        query_vector: list[float],
        top_k:        int,
        tenant_id:    str = "default",
    ) -> list[Chunk]:
        """Search all levels simultaneously — results may come from any level."""
        if self._faiss_index is None:
            return []

        import faiss
        import numpy as np

        query = np.array([query_vector], dtype="float32")
        distances, indices = self._faiss_index.search(query, top_k * 2)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            node = self._all_nodes[idx]
            if node.tenant_id != tenant_id:
                continue
            node.metadata["score"]      = float(dist)
            node.metadata["score_type"] = "raptor"
            results.append(node)
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Tree building internals
    # ------------------------------------------------------------------

    def _build_level(
        self,
        nodes: list[Chunk],
        level: int,
    ) -> list[Chunk]:
        """Cluster nodes, summarize each cluster, return summary chunks."""
        from sklearn.cluster import KMeans

        n_clusters = min(self.n_clusters, len(nodes) // self.min_cluster_size)
        if n_clusters < 1:
            return []

        vectors = np.array([n.embedding for n in nodes], dtype="float32")

        # K-Means clustering on the embedding space
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        summary_nodes = []
        for cluster_id in range(n_clusters):
            members = [nodes[i] for i, l in enumerate(labels) if l == cluster_id]
            if len(members) < 1:
                continue

            # Build passage text for summarization
            passages = "\n\n---\n\n".join(m.content for m in members)

            # Summarize with LLM
            summary_text = self._summarize(passages)
            if not summary_text:
                continue

            # Embed the summary
            summary_vec = self.embedder.embed_query(summary_text)

            # Create summary chunk with provenance metadata
            summary_chunk = Chunk(
                chunk_id    = str(uuid.uuid4()),
                doc_id      = members[0].doc_id,  # inherit from first member
                content     = summary_text,
                embedding   = summary_vec,
                tenant_id   = members[0].tenant_id,
                metadata    = {
                    "raptor_level":     level,
                    "raptor_node_type": "summary",
                    "raptor_cluster":   cluster_id,
                    "raptor_children":  [m.chunk_id for m in members],
                    "raptor_n_members": len(members),
                },
            )
            summary_nodes.append(summary_chunk)

        return summary_nodes

    def _summarize(self, passages: str) -> str:
        """Direct API call for summarization — bypasses RAG prompt builder."""
        import os, json, requests
        from dotenv import load_dotenv
        load_dotenv(override=False)

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return ""

        prompt = (
            "Summarize the following text passages into one coherent paragraph. "
            "Preserve all key facts and technical details.\n\n"
            f"{passages}\n\nSummary:"
        )

        try:
            resp = requests.post(
                url     = "https://openrouter.ai/api/v1/chat/completions",
                headers = {"Authorization": f"Bearer {api_key}",
                           "Content-Type": "application/json"},
                data    = json.dumps({
                    "model":       "openrouter/free",
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  300,
                    "temperature": 0.0,
                }),
                timeout = 60,
            )
            resp.raise_for_status()
            data    = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            return content.strip() if content else ""
        except Exception as e:
            print(f"  [RAPTOR] Summarization failed: {e}")
            return ""

    def _build_faiss_index(self) -> None:
        import faiss
        import numpy as np

        if not self._all_nodes or self._dimension is None:
            return

        vectors = np.array(
            [n.embedding for n in self._all_nodes],
            dtype="float32"
        )
        self._faiss_index = faiss.IndexFlatIP(self._dimension)
        self._faiss_index.add(vectors)

    def tree_stats(self) -> dict:
        if not self._all_nodes:
            return {}
        by_level: dict[int, int] = {}
        for node in self._all_nodes:
            lvl = node.metadata.get("raptor_level", 0)
            by_level[lvl] = by_level.get(lvl, 0) + 1
        return {
            "total_nodes": len(self._all_nodes),
            "by_level":    dict(sorted(by_level.items())),
        }