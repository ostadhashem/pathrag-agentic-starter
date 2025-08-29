from pathrag.agents.autogen_app import build_agents

if __name__ == "__main__":
    user, planner = build_agents()
    image_path = "sample_he.png"
    question = "What are a few well-developed cell nests with?"

    task = f'''
    Step 1: Call histo(image_path="{image_path}")
    Step 2: Call vlm_full(image_path="{image_path}", question="{question}")
    Step 3: For each returned patch, call vlm_patch(patch_id=<id>, question="{question}")
    Step 4: Aggregate all candidates (full + patches) and call reason(question="{question}", candidates=[...]).
    RETURN ONLY the final fused answer.
    '''
    result = user.initiate_chat(planner, message=task)
    print("FINAL ANSWER:", result.summary)