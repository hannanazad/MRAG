"""Qwen2.5-VL-7B-Instruct (default) wrapper.

Loads the explicit `Qwen2_5_VLForConditionalGeneration` + `AutoProcessor` +
`qwen_vl_utils.process_vision_info` path. Falls back to the 3B model if 7B
doesn't fit on the available GPU.

The prompt mirrors MUTCD's own taxonomy: outputs are blocked by rule type
(Standards / Guidance / Options / Support) and citations are restricted to
an explicit whitelist constructed from retrieval results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("mrag.vlm")


class VLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        fallback_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.model_name = model_name
        self.fallback_name = fallback_name
        self.torch_dtype = torch_dtype
        self._model = None
        self._processor = None
        self._loaded_name = None

    def load(self) -> "VLM":
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        try:
            self._do_load(self.model_name, torch_dtype=getattr(torch, self.torch_dtype))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            log.warning("VLM %s OOM/fail (%r); falling back to %s",
                        self.model_name, e, self.fallback_name)
            torch.cuda.empty_cache()
            self._do_load(self.fallback_name, torch_dtype=getattr(torch, self.torch_dtype))
        return self

    def _do_load(self, name, torch_dtype):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch_dtype, device_map="auto",
        ).eval()
        self._processor = AutoProcessor.from_pretrained(name)
        self._loaded_name = name
        log.info("VLM loaded: %s (%s)", name, torch_dtype)

    @property
    def loaded_name(self) -> str:
        return self._loaded_name or ""

    # ----- generation ------------------------------------------------------

    def answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        figures: List[Dict[str, Any]],
        pages: List[Dict[str, Any]],
        max_new_tokens: int = 480,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info
        messages = self._build_messages(question, chunks, figures, pages)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        out_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        return out_text

    # ----- prompt ----------------------------------------------------------

    def _build_messages(self, question, chunks, figures, pages):
        from pathlib import Path
        content: List[Dict[str, Any]] = []

        # Attach figure / page images (in order; they'll be cited as [Image N]).
        used_visuals = []
        for f in figures:
            ip = f.get("image_path", "")
            if ip and Path(ip).exists():
                content.append({"type": "image", "image": f"file://{ip}"})
                used_visuals.append(("Figure", f))
        for p in pages:
            ip = p.get("image_path", "")
            if ip and Path(ip).exists():
                content.append({"type": "image", "image": f"file://{ip}"})
                used_visuals.append(("Page",   p))

        # Group chunks by content_type so the model can mirror MUTCD's taxonomy.
        groups: Dict[str, List[Dict[str, Any]]] = {
            "Standard": [], "Guidance": [], "Option": [], "Support": [],
        }
        for c in chunks:
            ct = c.get("content_type", "Support")
            groups.setdefault(ct, []).append(c)

        # Build the textual evidence section, type-by-type.
        evidence_blocks = []
        for ct in ("Standard", "Guidance", "Option", "Support"):
            cs = groups.get(ct, [])
            if not cs: continue
            evidence_blocks.append(f"=== {ct} provisions ===")
            for c in cs:
                evidence_blocks.append(
                    f"[Section {c.get('section_id')} §{c.get('ordinal')} — "
                    f"{c.get('section_title','')} (p.{c.get('page_printed','?')})]\n"
                    f"{(c.get('text','') or '')[:1400]}"
                )
            evidence_blocks.append("")

        # Visual-evidence index
        visual_lines = []
        for i, (kind, v) in enumerate(used_visuals, 1):
            if kind == "Figure":
                visual_lines.append(
                    f"[Image {i}] {v.get('figure_id','?')} (p.{v.get('page_printed','?')}): "
                    f"{(v.get('caption','') or '')[:160]}"
                )
            else:
                visual_lines.append(
                    f"[Image {i}] Page {v.get('page_printed','?')} (full page view)"
                )

        # Allowed-citation whitelist
        allowed_cites = []
        for c in chunks:
            allowed_cites.append(
                f"Section {c.get('section_id')} {c.get('content_type')} §{c.get('ordinal')} (p.{c.get('page_printed','?')})"
            )
        for kind, v in used_visuals:
            if kind == "Figure":
                allowed_cites.append(f"{v.get('figure_id','?')} (p.{v.get('page_printed','?')})")
            else:
                allowed_cites.append(f"Page {v.get('page_printed','?')}")

        prompt = (
            "You are an expert reader of the Manual on Uniform Traffic Control Devices "
            "(MUTCD, 11th Edition). Answer the user's question using ONLY the evidence below.\n\n"
            "MUTCD distinguishes four normative categories. Treat them carefully:\n"
            "  - Standard: MANDATORY requirements (modal verb: shall).\n"
            "  - Guidance: RECOMMENDED practice (modal verb: should).\n"
            "  - Option: PERMITTED practice (modal verb: may).\n"
            "  - Support: explanatory or informational only — never normative.\n\n"
            "Output format (use these exact section headings, omit any that have no content):\n"
            "  Direct Answer: 2–3 sentences in plain language.\n"
            "  Standards (mandatory): bullets quoting the relevant Standard provision(s).\n"
            "  Guidance (recommended): bullets.\n"
            "  Options (permitted): bullets.\n"
            "  Visual evidence: one sentence per relevant image, referenced as [Image N].\n"
            "  Citations: bullets, ONE PER LINE, chosen ONLY from the allowed list below.\n\n"
            "Rules:\n"
            "  - Never invent section numbers, figure numbers, or page numbers.\n"
            "  - If the evidence is insufficient, say so plainly and stop.\n"
            "  - Quote MUTCD wording verbatim when stating a Standard provision.\n\n"
            f"Question: {question}\n\n"
            f"Visual evidence ({len(visual_lines)} images attached):\n"
            + ("\n".join(visual_lines) if visual_lines else "(none)") + "\n\n"
            f"Text evidence:\n" + "\n".join(evidence_blocks) + "\n"
            f"Allowed citations (use ONLY these strings verbatim):\n"
            + "\n".join(f"  - {c}" for c in allowed_cites) + "\n"
        )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]
