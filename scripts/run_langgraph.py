from pathrag.agents.langgraph_app import build_graph
if __name__ == "__main__":
    app = build_graph()
    init = {"image_path": "sample_he.png","question": "What are a few well-developed cell nests with?","mode":"answer","top_k":3}
    last = None
    for s in app.stream(init): last = s
    print("FINAL ANSWER:", last["pathrag"]["result"]["final_answer"])
    print("PATCHES:", last["pathrag"]["result"]["patches"])
    print("QUERIES:", last["pathrag"]["result"]["queries"])