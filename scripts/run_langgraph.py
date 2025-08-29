from pathrag.agents.langgraph_app import build_graph

if __name__ == "__main__":
    app = build_graph()
    init = {
        "image_path": "sample_he.png",
        "question": "What are a few well-developed cell nests with?",
        "is_he": False, "patches": [], "candidates": [], "final_answer": ""
    }
    last_state = None
    for s in app.stream(init):
        last_state = s
    # 'reason' is the final node key
    print("FINAL ANSWER:", last_state["reason"]["final_answer"])