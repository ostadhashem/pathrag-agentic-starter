from typing import List, Dict, Any
import os

# --- Mocked domain tools. Replace with your real implementations. ---

def run_histocartography(image_path: str) -> Dict[str, Any]:
    """Simulate stain normalization + nuclei detection + patch selection."""
    # TODO: Replace with real HistoCartography pipeline or REST call.
    return {
        "is_he": True,
        "patches": [f"{image_path}::patchA", f"{image_path}::patchB", f"{image_path}::patchC"]
    }

def run_llava_med_on_image(image_path: str, question: str) -> str:
    """Simulate LLaVA-Med caption/answer for the whole image."""
    # TODO: Replace with your VLM inference (local or remote).
    return f"[IMG:{image_path}] candidate for: {question}"

def run_llava_med_on_patch(patch_id: str, question: str) -> str:
    """Simulate LLaVA-Med caption/answer for a patch."""
    # TODO: Replace with your VLM inference (local or remote).
    return f"[PATCH:{patch_id}] candidate for: {question}"

def gpt_reason(question: str, candidates: List[str]) -> str:
    """Fuse candidates using a text LLM (OpenAI by default)."""
    # Safe fallback that avoids failing without API key:
    api_key = os.environ.get("OPENAI_API_KEY")
    content = "You are a careful medical reasoning assistant.\n"
    content += f"Question: {question}\nCandidates:\n- " + "\n- ".join(candidates)
    if not api_key:
        # No key? Return a mocked answer.
        return "[mocked] keratinization (example)\n" + content
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[mocked due to error: {e}] keratinization (example)"