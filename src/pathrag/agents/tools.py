from __future__ import annotations
from typing import List, Dict, Any, Iterable, Tuple
import os, json
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from pathrag.utils.logging import get_logger
load_dotenv()
logger = get_logger("pathrag.tools")

NUCLEI_THRESHOLD = 5
GRID_SIZE = 3
OVERLAP = 0.20
DEFAULT_TOP_K = 3

OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", "src/pathrag/pipeline/files"))
(OUTPUT_ROOT / "query").mkdir(parents=True, exist_ok=True)

@dataclass
class PatchInfo:
    patch_id: str
    bbox: Tuple[int, int, int, int]
    nuclei_count: int

def _mock_nuclei_count(img: Image.Image) -> int:
    w, h = img.size
    return 10 if (w * h) > 64*64 else 0

def _grid_with_overlap(w: int, h: int, grid: int = GRID_SIZE, overlap: float = OVERLAP):
    step_x = int(w / grid); step_y = int(h / grid)
    dx = int(step_x * (1 + overlap)); dy = int(step_y * (1 + overlap))
    for gy in range(grid):
        for gx in range(grid):
            x0 = max(0, gx*step_x - int(step_x*overlap/2))
            y0 = max(0, gy*step_y - int(step_y*overlap/2))
            x1 = min(w, x0 + dx); y1 = min(h, y0 + dy)
            yield (x0,y0,x1,y1)

def run_histocartography(image_path: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    logger.info(f"Histo start: image={image_path}, top_k={top_k}")
    p = Path(image_path)
    img = Image.new("RGB", (512, 512)) if not p.exists() else Image.open(p).convert("RGB")
    nuclei_total = _mock_nuclei_count(img)
    is_he = nuclei_total >= NUCLEI_THRESHOLD
    patches: List[PatchInfo] = []
    for idx, bbox in enumerate(_grid_with_overlap(*img.size)):
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        nuc = int(area // (64*64))
        patches.append(PatchInfo(patch_id=f"{p.name}::P{idx}", bbox=bbox, nuclei_count=nuc))
    patches.sort(key=lambda x: x.nuclei_count, reverse=True)
    selected = patches[:top_k] if is_he else []
    logger.info(f"Histo result: is_he={is_he}, selected_patches={len(selected)}")
    return {"is_he": is_he, "patches": [pi.__dict__ for pi in selected]}

def make_llava_query_files(image_path: str, question: str, patch_infos: List[PatchInfo]) -> Dict[str, str]:
    logger.info(f"Writing LLaVA queries for image={image_path}, patches={len(patch_infos)}")
    def wjsonl(path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False)+"\n")
    qroot = OUTPUT_ROOT / "query"
    files = {
        "image_direct": qroot / "image_direct.jsonl",
        "patch_direct": qroot / "patch_direct.jsonl",
        "image_description": qroot / "image_description.jsonl",
        "patch_description": qroot / "patch_description.jsonl",
    }
    image_rows_direct = [{"image": image_path, "text": question, "question_id": 0}]
    image_rows_desc   = [{"image": image_path, "text": "Describe the following image in detail.", "question_id": 0}]
    patch_rows_direct = [{"image": image_path, "bbox": pi.bbox, "text": question, "question_id": 0} for pi in patch_infos]
    patch_rows_desc   = [{"image": image_path, "bbox": pi.bbox, "text": "Describe the following image in detail.", "question_id": 0} for pi in patch_infos]
    wjsonl(files["image_direct"], image_rows_direct)
    wjsonl(files["image_description"], image_rows_desc)
    wjsonl(files["patch_direct"], patch_rows_direct)
    wjsonl(files["patch_description"], patch_rows_desc)
    return {k: str(v) for k, v in files.items()}

def _fusion_prompt_answer(question: str, full: List[str], patches: List[str]) -> str:
    lines = ["You are a professional pathologist.",
             f"• Perspective 1 (full): {full[0] if full else ''}"]
    for i,a in enumerate(patches, start=2): lines.append(f"• Perspective {i} (patch): {a}")
    lines.append(f"• Question: {question}")
    return "\\n".join(lines)

def _fusion_prompt_desc(question: str, desc_full: str, desc_patches: List[str]) -> str:
    lines = ["You are a professional pathologist.",
             f"• Description of image: {desc_full}"]
    for i,d in enumerate(desc_patches, start=1): lines.append(f"• Description of patch {i}: {d}")
    lines.append(f"• Question: {question}")
    return "\\n".join(lines)

def gpt_reason(question: str, mode: str, full_outputs: List[str], patch_outputs: List[str]) -> str:
    logger.info(f"GPT fusion: mode={mode}, full={len(full_outputs)}, patches={len(patch_outputs)}")
    content = _fusion_prompt_answer(question, full_outputs, patch_outputs) if mode == "answer" \
              else _fusion_prompt_desc(question, full_outputs[0] if full_outputs else "", patch_outputs)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "[mocked GPT] keratinization (example)\\n" + content
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":content}], temperature=0)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[mocked due to error: {e}] keratinization (example)"

def prepare_and_run(image_path: str, question: str, top_k: int = DEFAULT_TOP_K, mode: str = "answer") -> Dict[str, Any]:
    logger.info(f"Pipeline start: image={image_path}, mode={mode}, top_k={top_k}")
    info = run_histocartography(image_path, top_k=top_k)
    patches = [PatchInfo(**p) for p in info["patches"]]
    files = make_llava_query_files(image_path, question, patches)
    # mock VLM outputs (you can wire real LLaVA-Med later)
    img_ans = [f"[mock IMAGE-ans] {question}"] if mode == "answer" else [f"[mock IMAGE-desc] image description"]
    patch_ans = [f"[mock PATCH-{i}] ..." for i,_ in enumerate(patches,1)]
    final = gpt_reason(question, mode, img_ans, patch_ans)
    logger.info("Pipeline done.")
    return {"is_he": info["is_he"], "patches": [p.__dict__ for p in patches], "queries": files, "final_answer": final}
