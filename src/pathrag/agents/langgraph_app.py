from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pathrag.agents.tools import prepare_and_run
from pathrag.utils.logging import get_logger

class PathRAGState(TypedDict):
    image_path: str
    question: str
    mode: str
    top_k: int
    result: Dict[str, Any]

logger = get_logger("pathrag.langgraph")

def node_pipeline(state: PathRAGState):
    logger.info(f"LangGraph node start: {state}")
    state["result"] = prepare_and_run(
        image_path=state["image_path"],
        question=state["question"],
        top_k=state.get("top_k", 3),
        mode=state.get("mode", "answer"),
    )
    logger.info("LangGraph node end.")
    return state

def build_graph():
    graph = StateGraph(PathRAGState)
    graph.add_node("pathrag", node_pipeline)
    graph.add_edge(START, "pathrag")
    graph.add_edge("pathrag", END)
    return graph.compile(checkpointer=MemorySaver())
