from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pathrag.agents.tools import (
    run_histocartography,
    run_llava_med_on_image,
    run_llava_med_on_patch,
    gpt_reason,
)

class PathRAGState(TypedDict):
    image_path: str
    question: str
    is_he: bool
    patches: List[str]
    candidates: List[str]
    final_answer: str

def node_detect(state: PathRAGState):
    info = run_histocartography(state["image_path"])
    state["is_he"] = bool(info.get("is_he", False))
    state["patches"] = info.get("patches", []) if state["is_he"] else []
    return state

def node_vlm(state: PathRAGState):
    q = state["question"]
    cand = [run_llava_med_on_image(state["image_path"], q)]
    for p in state["patches"]:
        cand.append(run_llava_med_on_patch(p, q))
    state["candidates"] = cand
    return state

def node_reason(state: PathRAGState):
    state["final_answer"] = gpt_reason(state["question"], state["candidates"])
    return state

def build_graph():
    graph = StateGraph(PathRAGState)
    graph.add_node("detect", node_detect)
    graph.add_node("vlm", node_vlm)
    graph.add_node("reason", node_reason)

    graph.add_edge(START, "detect")
    graph.add_edge("detect", "vlm")
    graph.add_edge("vlm", "reason")
    graph.add_edge("reason", END)

    return graph.compile(checkpointer=MemorySaver())