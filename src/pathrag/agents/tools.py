"""
Module: agents.tools

Purpose:
- Define small typed domain objects (e.g., Patch) used across LangGraph nodes.
- Provide typed interfaces for tiling, ranking, retrieval, per-patch agents, critique, and final fusion.
- Keep annotations lazy to avoid import cycles and speed up startup.

Notes:
- Annotations are postponed (strings). Use typing.get_type_hints(...) if runtime resolution is needed.
- Prefer clear type hints (List[Patch], Dict[str, Any]) to improve IDE support and team readability.
- Use @dataclass for compact, maintainable domain models.
"""
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

# =========================
# STAGE 1: TILING + HC RANK
# =========================
def tile_image(image_path: str, tile_size: int = 224) -> List[Patch]:
    """Split the input image into a regular grid of tiles (no GPU required).

    Design:
      - Keeps IDs deterministic (P0, P1, …) based on scan order.
      - Does NOT load image pixels here if you already have a tiler upstream; feel free to replace
        with your WSI tiler (OpenSlide, tifffile) that yields (bbox) without loading full image.

    Args:
        image_path: Path to the WSI or large image.
        tile_size: Width/height of square tiles (pixels).
        stride: Optional stride; if None, defaults to tile_size (no overlap).
                Use a smaller stride (< tile_size) to introduce overlap.

    Returns:
        A dense list of Patch with zero scores (to be ranked in the next step).
    """
    # TODO (real impl):
    #   - Use OpenSlide/pyvips/tifffile to stream tiles without loading the full WSI.
    #   - Compute (W, H) from image metadata; iterate y,x over range(0, H, stride).
    # MOCK (minimal): return 9 tiles of 224×224 from the top-left quadrant
    tiles: List[Patch] = []
    k = 0
    grid = 3  # 3×3 as a lightweight demo
    step = tile_size
    for i in range(grid):
        for j in range(grid):
            bbox = (j*step, i*step, j*step + tile_size, i*step + tile_size)
            tiles.append(Patch(id=f"P{k}", bbox=bbox, score=0.0))
            k += 1
    logger.info(f"Tiled image {image_path} into {len(tiles)} patches.")
    return tiles


def histocartography_rank(image_path: str, patches: List[Patch]) -> List[Patch]:
    """Rank patches by a HistoCartography-derived proxy (e.g., nuclei density).

    Contract:
      - SAME length/order as input is NOT required; we return a sorted list (desc by score).
      - Score meaning: larger == more informative (for H&E).

    Replace this mock with:
      - Your nuclei segmentation + counting per patch, or
      - Any HC embedding → importance scorer (e.g., graph centrality).

    Args:
        image_path: Path to image (used if you crop & analyze pixels here).
        patches: Dense grid from `tile_image`.

    Returns:
        New list of Patch with `.score` filled, sorted descending by score.
    """
    # TODO (real impl):
    #   for p in patches:
    #       crop = read_crop(image_path, p.bbox)
    #       score = nuclei_count(crop) or nuclei_density(crop)
    #       out.append(Patch(p.id, p.bbox, float(score)))
    # MOCK: monotonically decreasing scores
    ranked: List[Patch] = []
    for i, p in enumerate(patches):
        ranked.append(Patch(id=p.id, bbox=p.bbox, score=1.0 - 0.05 * i))
    ranked.sort(key=lambda q: q.score, reverse=True)
    logger.info(f"Ranked {len(ranked)} patches by HistoCartography proxy.")
    return ranked

# ==============================
# STAGE 2: CHEIF + COMMON PATCH
# ==============================
def cheif_rank(image_path: str, patches: List[Patch]) -> List[Patch]:
    """Rank patches by CHEIF attention (foundation model signal).

    Replace this mock with:
      - A call to your CHEIF model to produce per-patch attentions/saliency.
      - Normalize scores to [0,1] if you mix with other sources.

    Returns:
        List[Patch] sorted descending by attention score.
    """
    # TODO (real impl):
    #   att = cheif_attention(image_path, [p.bbox for p in patches]) -> List[float]
    #   return sorted([Patch(p.id, p.bbox, att[i]) ...], key=lambda x: x.score, reverse=True)
    ranked: List[Patch] = []
    for i, p in enumerate(patches):
        ranked.append(Patch(id=p.id, bbox=p.bbox, score=1.0 - 0.03 * i))
    ranked.sort(key=lambda q: q.score, reverse=True)
    return ranked


def common_patches(hc: List[Patch], cheif: List[Patch], top_k: int = 6) -> List[Patch]:
    """Intersect/merge HC and CHEIF rankings into a consensus Top-K.

    Strategy (simple & effective):
      1) Take a wider band from each list (e.g., 2×top_k) to avoid missing near-misses.
      2) Merge by patch.id, accumulate scores from both sources.
      3) Aggregate scores (mean) → sort desc → take Top-K.

    Notes:
      - If HC/CHEIF disagree on bbox (shouldn’t if tiling is shared), prefer the first occurrence.
      - You can switch to rank-based fusion (e.g., Borda count) if score scales differ a lot.

    Args:
        hc: HC-ranked patches (desc by HC score).
        cheif: CHEIF-ranked patches (desc by attention).
        top_k: Output size after fusion.

    Returns:
        Top-K fused Patch list (desc by aggregated score).
    """
    band = top_k * 2
    by_id: Dict[str, Tuple[Tuple[int,int,int,int], List[float]]] = {}

    def add(lst: List[Patch]):
        for p in lst[:band]:
            if p.id not in by_id:
                by_id[p.id] = (p.bbox, [p.score])
            else:
                by_id[p.id][1].append(p.score)

    add(hc)
    add(cheif)

    fused: List[Patch] = []
    for pid, (bbox, scores) in by_id.items():
        avg = sum(scores) / len(scores)   # simple average of available scores
        fused.append(Patch(id=pid, bbox=bbox, score=avg))

    fused.sort(key=lambda q: q.score, reverse=True)
    logger.info(f"Fused {len(fused)} patches from HC and CHEIF into Top-{top_k}.")
    return fused[:top_k]

# ===============================
# STAGE 3: LABELING + RETRIEVAL
# ===============================
def identify_subpathology(image_path: str) -> str:
    """Predict a coarse tissue/site/sub-pathology label from the image.

    Replace this mock with your actual classifier (e.g., PLIP/CLIP head, UTSW model).
    Keep the return value SHORT and canonicalized (used as a retrieval query/key).

    Args:
        image_path: path to the source image (or WSI).

    Returns:
        A normalized label string (e.g., "squamous_cell_carcinoma").
    """
    # TODO (real impl):
    #   label = plip_or_clip_predict(image_path)
    #   return normalize(label)
    return "squamous_cell_carcinoma?"


def retrieve_subpath_captions(label: str, top_m: int = 5) -> list[str]:
    """Fetch top-M textual snippets/captions relevant to the label.

    Replace this mock with your retriever (BiomedCLIP/PLIP embeddings + FAISS).
    Normalize text (strip, de-dupe) and keep snippets concise.

    Args:
        label: normalized label from identify_subpathology.
        top_m: how many captions to return.

    Returns:
        List of short strings (captions/knowledge to condition later agents).
    """
    # TODO (real impl):
    #   qvec = embed(label)
    #   results = faiss.search(qvec, top_m)
    #   return [r.text for r in results]
    return [f"{label} caption #{i+1}" for i in range(top_m)]


# =======================================
# STAGE 4: ROI + PATCH AGENTS (per patch)
# =======================================
def roi_agent_describe(patch: Patch, question: str) -> dict[str, object]:
    """Return whether the patch is useful and a brief clinical description.

    Replace this mock with a VLM (image) or LLM (text-only) agent.
    The question is provided so the ROI can tailor its description.

    Returns:
        {"useful": bool, "description": str}
    """
    # TODO (real impl):
    #   crop = load_crop(patch.bbox)
    #   text = vlm_describe(crop, question)
    #   useful = detector_or_rule(text)
    return {"useful": True, "description": f"ROI {patch.id}: keratin pearls likely."}


def patch_agent_contribution(patch: Patch, question: str, full_captions: list[str]) -> str:
    """Explain how this patch contributes to answering the question.

    Provide one focused sentence (ideal for later fusion).
    You can include a brief reference to the most relevant caption.

    Returns:
        A short string (<= 1–2 lines).
    """
    # TODO (real impl):
    #   crop = load_crop(patch.bbox)
    #   return vlm_or_llm_contribution(crop, question, full_captions)
    cap = full_captions[0] if full_captions else "domain context"
    return f"{patch.id} supports the answer via {cap}."

# =====================================
# STAGE 6: QUESTION-AWARE RE-RANK TOPK
# =====================================
def rerank_for_question(
    patches: list[Patch],
    summaries: list[str],
    question: str,
    k: int,
) -> list[int]:
    """Return indices (into `patches`) for the Top-K most relevant summaries.

    Replace this mock with:
      - Embedding similarity (question vs summary), or
      - A cross-encoder scoring model.

    NOTE: Return INDICES (ints), not Patch objects, to avoid state duplication.
    """
    # TODO (real impl):
    #   scores = cross_encoder(question, summaries)
    #   idx = sorted(range(len(summaries)), key=lambda i: scores[i], reverse=True)[:k]
    return list(range(min(k, len(patches))))

# ============================
# STAGE 7: FINAL FUSION (LLM)
# ============================
def fuse_answer(
    question: str,
    label: str,
    chosen: list[tuple[Patch, str]],
    mode: str = "answer",     # "answer" | "description"
) -> str:
    """Fuse per-patch statements + label into the final answer.

    Replace this with a GPT-class call if OPENAI_API_KEY is set.
    Keep the output short and evidence-grounded.

    Args:
        chosen: list of (Patch, summary) for Top-K after re-ranking.
    """
    # TODO (real impl): construct a careful prompt; include 1–2 safety rules.
    bullets = "\n".join([f"- {p.id} → {s}" for p, s in chosen])
    return f"[mock {mode}] Q: {question}\nlabel={label}\n{bullets}"
