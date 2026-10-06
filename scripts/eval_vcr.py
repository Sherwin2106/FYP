#!/usr/bin/env python
"""Zero-shot Phase I baseline on the VCR benchmark (Q->A, QA->R, Q->AR accuracy).

    python scripts/eval_vcr.py --vcr-root /data/vcr1 --split val --limit 500
    python scripts/eval_vcr.py --vcr-root /data/vcr1 --limit 500 --commonsense     # + ConceptNet context

The numbers are the reference point that the Phase II fine-tuned SmolVLM should beat.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from svcr.config import load_config  # noqa: E402
from svcr.conceptnet import CommonsenseEnricher, ConceptNetClient  # noqa: E402
from svcr.preprocessing import Preprocessor  # noqa: E402
from svcr.schemas import DraftReasoning, VisualConcepts  # noqa: E402
from svcr.taxonomy import Taxonomy  # noqa: E402
from svcr.vcr_dataset import draw_tags, load_vcr  # noqa: E402
from svcr.vlm import SmolVLMEngine  # noqa: E402

TAG_NOTE = "People and objects mentioned in the text are marked in the image with labelled boxes."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcr-root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--commonsense", action="store_true", help="add ConceptNet facts to the prompt")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "vcr_eval.jsonl"))
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(message)s")
    pre = Preprocessor(cfg.preprocess)
    vlm = SmolVLMEngine(cfg.vlm)
    enricher = None
    if args.commonsense:
        tax = Taxonomy.load(cfg.resolve_path(cfg.reasoning.taxonomy_path))
        enricher = CommonsenseEnricher(ConceptNetClient.from_config(cfg.conceptnet), tax, cfg.conceptnet)

    n = qa = qar = ar = 0
    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as log_fh:
        for item in load_vcr(args.vcr_root, args.split, args.limit):
            frame = pre.process_bgr(draw_tags(item), str(item.image_path))
            context = TAG_NOTE
            if enricher is not None:
                cs = enricher.enrich(VisualConcepts(people_count=2), DraftReasoning(activity=item.question),
                                     question=" ".join([item.question, *item.answer_choices]))
                if cs.facts:
                    context += "\nCommonsense knowledge:\n" + "\n".join(f"- {f.sentence()}." for f in cs.facts[:8])

            p_a = vlm.score_options(frame, item.question, item.answer_choices, context)
            pred_a = int(p_a.argmax())
            rec = {"annot_id": item.annot_id, "pred_answer": pred_a, "answer_label": item.answer_label,
                   "p_answer": [round(float(x), 4) for x in p_a]}
            if item.rationale_choices:
                gold_answer = item.answer_choices[item.answer_label if item.answer_label is not None else pred_a]
                q_r = f"{item.question} The answer is: {gold_answer} Why is this answer correct?"
                p_r = vlm.score_options(frame, q_r, item.rationale_choices, context)
                rec.update(pred_rationale=int(p_r.argmax()), rationale_label=item.rationale_label)

            if item.answer_label is not None:
                n += 1
                ok_a = pred_a == item.answer_label
                ok_r = rec.get("pred_rationale") == item.rationale_label
                qa += ok_a
                ar += ok_r
                qar += ok_a and ok_r
                if n % 10 == 0:
                    print(f"[{n}] Q->A {qa / n:.1%}  QA->R {ar / n:.1%}  Q->AR {qar / n:.1%}  "
                          f"({(time.time() - t0) / n:.1f}s/item)")
            log_fh.write(json.dumps(rec) + "\n")

    if n:
        summary = {"items": n, "Q->A": qa / n, "QA->R": ar / n, "Q->AR": qar / n,
                   "model": cfg.vlm.model_id, "commonsense": args.commonsense, "chance": {"Q->A": 0.25, "Q->AR": 0.0625}}
        print(json.dumps(summary, indent=2))
        Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
