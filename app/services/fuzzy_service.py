from thefuzz import fuzz


SIMILARITY_THRESHOLD = 90


def _normalize(name: str) -> str:
    return " ".join(name.casefold().split())


def fuzzy_match_names(
    extracted_names: list[str],
    query_names: list[dict],
) -> list[dict]:
    """Perform fuzzy matching between extracted and query names.

    Scores whole names with token_sort_ratio (typo-tolerant, insensitive to
    word order and case). partial_ratio is deliberately not used: it scores a
    short query as a substring match, so "Jo Sm" would match "John Smith".
    """
    matches = []

    for query in query_names:
        query_full = f"{query['first_name']} {query['last_name']}"

        best_match = None
        best_score = 0

        for extracted in extracted_names:
            score = fuzz.token_sort_ratio(_normalize(query_full), _normalize(extracted))

            if score > best_score:
                best_score = score
                best_match = extracted

        if best_score >= SIMILARITY_THRESHOLD:
            matches.append({
                "extracted_name": best_match,
                "matched_name": query_full,
                "score": best_score / 100.0,
            })

    return matches
