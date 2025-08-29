import os
from typing import List
from autogen import AssistantAgent, UserProxyAgent, tool
from pathrag.agents.tools import (
    run_histocartography,
    run_llava_med_on_image,
    run_llava_med_on_patch,
    gpt_reason,
)

# Wrap domain tools as AutoGen @tool so the planner can call them.

@tool
def histo(image_path: str) -> dict:
    """Return whether H&E and list of top patches."""
    return run_histocartography(image_path)

@tool
def vlm_full(image_path: str, question: str) -> str:
    """Run VLM on the full image."""
    return run_llava_med_on_image(image_path, question)

@tool
def vlm_patch(patch_id: str, question: str) -> str:
    """Run VLM on a single patch."""
    return run_llava_med_on_patch(patch_id, question)

@tool
def reason(question: str, candidates: List[str]) -> str:
    """Fuse candidates with a text reasoner."""
    return gpt_reason(question, candidates)

def build_agents():
    planner = AssistantAgent(
        "planner",
        system_message="You plan Path-RAG steps and call tools.",
        llm_config={
            "config_list": [
                {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}
            ],
            "temperature": 0
        },
        tools=[histo, vlm_full, vlm_patch, reason],
    )
    user = UserProxyAgent("user")
    return user, planner