"""Candidate embedding models and how each must be prompted (per the model authors)."""

from dataclasses import dataclass

BGE_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_name: str
    query_prefix: str = ""      # retrieval: question side
    passage_prefix: str = ""    # retrieval: chunk side
    symmetric_prefix: str = ""  # sentence-vs-sentence similarity (pairs/triplets)
    trust_remote_code: bool = False


MODELS = {m.id: m for m in [
    ModelSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    ModelSpec("mpnet", "sentence-transformers/all-mpnet-base-v2"),
    ModelSpec("bge-small", "BAAI/bge-small-en-v1.5", query_prefix=BGE_QUERY),
    ModelSpec("bge-base", "BAAI/bge-base-en-v1.5", query_prefix=BGE_QUERY),
    # e5: "query: " on both sides for symmetric tasks, query/passage for retrieval.
    ModelSpec("e5-base", "intfloat/e5-base-v2", query_prefix="query: ", passage_prefix="passage: ",
              symmetric_prefix="query: "),
    # nomic: task prefixes are mandatory; same prefix on both sides for symmetric similarity.
    ModelSpec("nomic", "nomic-ai/nomic-embed-text-v1.5", query_prefix="search_query: ",
              passage_prefix="search_document: ", symmetric_prefix="search_query: ", trust_remote_code=True),
    # Same model, Nomic's recommended prefix for similarity/clustering tasks on the symmetric side.
    ModelSpec("nomic-clustering", "nomic-ai/nomic-embed-text-v1.5", query_prefix="search_query: ",
              passage_prefix="search_document: ", symmetric_prefix="clustering: ", trust_remote_code=True),
    ModelSpec("arctic-m", "Snowflake/snowflake-arctic-embed-m-v1.5", query_prefix=BGE_QUERY),
]}
