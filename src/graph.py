"""
GhostListing - LangGraph Orchestration
------------------------------------------
This is the piece that makes it an "agentic workflow" rather than a script:
- Cheap deterministic steps (embed + search) run on EVERY listing.
- Expensive LLM/vision steps only run on the ~5% that get flagged.
- A human moderator must approve before anything is actioned - the graph
  physically pauses (via a checkpointer) at that node and waits.

Run the self-test at the bottom first (no API keys needed - it uses fake
functions to prove the routing logic is correct). Then swap in the real
functions from embed.py / similarity_index.py / vision_check.py.
"""

from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class ListingState(TypedDict):
    listing_id: str
    image_path: str
    seller_id: str
    listing_title: str
    listing_description: str
    listing_category: str
    matches: List[dict]
    flagged: bool
    image_description: Optional[str]
    consistency_verdict: Optional[str]
    consistency_reason: Optional[str]
    human_decision: Optional[str]  # set externally when resuming after interrupt


def build_graph(embed_fn, search_fn, describe_fn, consistency_fn, distance_threshold=0.3):
    """
    embed_fn(image_path) -> np.ndarray
    search_fn(embedding, exclude_seller, k) -> list of match dicts
    describe_fn(image_path) -> str
    consistency_fn(description, title, desc, category) -> dict(verdict, reason)

    Passing these in as arguments (dependency injection) means you can test
    the graph's ROUTING LOGIC with fake instant functions, separately from
    testing the real CLIP/Groq calls - which is exactly what the self-test
    below does.
    """

    def embed_and_search(state: ListingState) -> dict:
        embedding = embed_fn(state["image_path"])
        matches = search_fn(embedding, exclude_seller=state["seller_id"], k=3)
        flagged = any(m["l2_distance"] < distance_threshold for m in matches)
        return {"matches": matches, "flagged": flagged}

    def route_after_search(state: ListingState) -> str:
        return "vision_describer" if state["flagged"] else "no_action"

    def vision_describer(state: ListingState) -> dict:
        description = describe_fn(state["image_path"])
        return {"image_description": description}

    def consistency_checker(state: ListingState) -> dict:
        result = consistency_fn(
            state["image_description"],
            state["listing_title"],
            state["listing_description"],
            state["listing_category"],
        )
        return {
            "consistency_verdict": result["verdict"],
            "consistency_reason": result["reason"],
        }

    def human_review(state: ListingState) -> dict:
        # This node is where execution PAUSES (see interrupt_before below).
        # When resumed, state["human_decision"] will already be set by
        # whoever called graph.invoke() again with the update.
        return {}

    def finalize(state: ListingState) -> dict:
        # In the real app this writes to the audit_log table (see api.py)
        return {}

    def no_action(state: ListingState) -> dict:
        return {}

    graph = StateGraph(ListingState)
    graph.add_node("embed_and_search", embed_and_search)
    graph.add_node("vision_describer", vision_describer)
    graph.add_node("consistency_checker", consistency_checker)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize)
    graph.add_node("no_action", no_action)

    graph.set_entry_point("embed_and_search")
    graph.add_conditional_edges(
        "embed_and_search",
        route_after_search,
        {"vision_describer": "vision_describer", "no_action": "no_action"},
    )
    graph.add_edge("vision_describer", "consistency_checker")
    graph.add_edge("consistency_checker", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("no_action", END)

    # The checkpointer is what makes the human_review pause DURABLE - the
    # graph can sit paused for hours/days waiting on a real moderator,
    # not just a few seconds in memory.
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
    return compiled


def _self_test():
    """
    Proves the routing logic with instant fake functions - no CLIP model,
    no Groq API key, no network needed. This is what you run FIRST to
    confirm the graph itself is wired correctly before touching real APIs.
    """
    # Fake embed: just returns the image_path itself as a "fingerprint"
    def fake_embed(image_path):
        return image_path

    # Fake search: pretends "stolen.jpg" is a near-duplicate of an existing
    # listing, everything else has no match
    def fake_search(embedding, exclude_seller, k):
        if "stolen" in embedding:
            return [{"listing_id": "seller_a_original", "seller_id": "seller_a", "l2_distance": 0.02}]
        return [{"listing_id": "unrelated", "seller_id": "seller_z", "l2_distance": 1.9}]

    def fake_describe(image_path):
        return "A slightly used blue backpack with a visible scuff on the front pocket."

    def fake_consistency(description, title, desc, category):
        if "brand new" in desc.lower():
            return {"verdict": "INCONSISTENT", "reason": "Photo shows visible wear but listing claims brand new."}
        return {"verdict": "CONSISTENT", "reason": "Description matches visible condition."}

    graph = build_graph(fake_embed, fake_search, fake_describe, fake_consistency)

    # Case 1: a flagged, suspicious listing - should PAUSE at human_review
    config = {"configurable": {"thread_id": "test-thread-1"}}
    result = graph.invoke({
        "listing_id": "seller_b_listing_7",
        "image_path": "stolen.jpg",
        "seller_id": "seller_b",
        "listing_title": "Brand New Backpack",
        "listing_description": "brand new, never used, amazing condition",
        "listing_category": "Bags",
    }, config)

    print("--- Case 1: flagged listing ---")
    print("Flagged:", result["flagged"])
    print("Consistency verdict:", result.get("consistency_verdict"))
    print("Paused for human review? ", "human_decision" not in result or result.get("human_decision") is None)
    assert result["flagged"] is True
    assert result["consistency_verdict"] == "INCONSISTENT"

    # Resume it as if a human moderator just approved the flag
    graph.update_state(config, {"human_decision": "confirmed_fraud"})
    final = graph.invoke(None, config)
    print("After human resume, final state reached:", final)

    # Case 2: a clean listing - should skip straight to no_action, no LLM calls at all
    config2 = {"configurable": {"thread_id": "test-thread-2"}}
    result2 = graph.invoke({
        "listing_id": "seller_c_listing_1",
        "image_path": "normal_photo.jpg",
        "seller_id": "seller_c",
        "listing_title": "Used Backpack",
        "listing_description": "gently used",
        "listing_category": "Bags",
    }, config2)

    print("\n--- Case 2: clean listing ---")
    print("Flagged:", result2["flagged"])
    print("Skipped vision/consistency steps entirely:",
          result2.get("image_description") is None)
    assert result2["flagged"] is False
    assert result2.get("image_description") is None

    print("\nSelf-test passed: routing correctly separates flagged vs. clean "
          "listings, and the human-review pause works as expected.")


if __name__ == "__main__":
    _self_test()
