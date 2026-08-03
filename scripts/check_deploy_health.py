"""
scripts/check_deploy_health.py

Read-only diagnostic for the contest retrieval path. Run it ON THE SERVER
(where MONGODB_URI is set) to see whether the slowness/crash fixes are
actually in effect:

    python scripts/check_deploy_health.py

It answers four questions:
  1. Does the Atlas Vector Search index exist and is it queryable?
     If not, every contest search takes the brute-force fallback.
  2. Does the (contest, year) btree index exist?
  3. How long do the hot metadata calls take, and is caching working?
  4. How long does a real semantic query take, and which path served it?

Writes nothing. Safe to run against production.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.db import get_db
from rag import contest_retriever as cr


def _ms(t0: float) -> str:
    return f"{(time.perf_counter() - t0) * 1000:.0f}ms"


def main() -> None:
    db = get_db()
    col = db[cr.COLLECTION_NAME]
    problems = []

    print("=" * 62)
    print("1. Atlas Vector Search index")
    print("=" * 62)
    try:
        indexes = list(col.list_search_indexes())
    except Exception as e:
        indexes = []
        print(f"   could not list search indexes: {e}")
    match = next((i for i in indexes if i["name"] == cr.VECTOR_INDEX_NAME), None)
    if match and match.get("queryable"):
        print(f"   OK  '{cr.VECTOR_INDEX_NAME}' exists and is queryable")
    elif match:
        print(f"   BUILDING  '{cr.VECTOR_INDEX_NAME}' status={match.get('status')} (not queryable yet)")
        problems.append("vector index still building - searches use the fallback until it finishes")
    else:
        print(f"   MISSING  no index named '{cr.VECTOR_INDEX_NAME}'")
        print(f"   found instead: {[i['name'] for i in indexes] or 'none'}")
        problems.append("vector index missing - run scripts/create_vector_index.py")

    print()
    print("=" * 62)
    print("2. (contest, year) btree index")
    print("=" * 62)
    names = col.index_information().keys()
    if any("contest" in n for n in names):
        print(f"   OK  present: {[n for n in names if 'contest' in n]}")
    else:
        print(f"   MISSING  indexes are: {list(names)}")
        problems.append("(contest, year) index missing - restart the app so ensure_indexes() runs")

    print()
    print("=" * 62)
    print("3. Hot metadata calls")
    print("=" * 62)
    cr.invalidate_meta_cache()
    t0 = time.perf_counter(); n = cr.collection_count(); cold_count = _ms(t0)
    t0 = time.perf_counter(); contests = cr.list_available_contests(); cold_list = _ms(t0)
    t0 = time.perf_counter(); cr.collection_count(); warm_count = _ms(t0)
    t0 = time.perf_counter(); cr.list_available_contests(); warm_list = _ms(t0)
    print(f"   documents in corpus     : {n}")
    print(f"   collection_count        : {cold_count} cold -> {warm_count} cached")
    print(f"   list_available_contests : {cold_list} cold -> {warm_list} cached")
    print(f"   contests indexed        : {[c['contest'] for c in contests]}")
    if not hasattr(cr, "invalidate_meta_cache"):
        problems.append("running OLD contest_retriever.py - the perf fixes are not deployed")

    print()
    print("=" * 62)
    print("4. Real semantic query")
    print("=" * 62)
    cr._warned_no_index = False
    t0 = time.perf_counter()
    results = cr.query("geometry triangle area", n_results=5)
    took = _ms(t0)
    served_by = "brute-force fallback" if cr._warned_no_index else "Atlas Vector Search"
    print(f"   returned {len(results)} results in {took}, served by: {served_by}")
    if results:
        top = results[0]
        print(f"   top hit: {top['contest']} {top['year']} "
              f"problem {top['problem_number']} (similarity {top.get('similarity')})")
    if cr._warned_no_index:
        problems.append("query ran through the brute-force fallback, not Atlas Vector Search")

    print()
    print("=" * 62)
    if problems:
        print(f"{len(problems)} PROBLEM(S) FOUND")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All checks passed - retrieval is using Atlas Vector Search with")
        print("indexes and caching in place.")
    print("=" * 62)


if __name__ == "__main__":
    main()
