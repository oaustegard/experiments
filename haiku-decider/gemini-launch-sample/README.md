# Gemini Decisions launch sample (2026-11-11)

60 random items each from PubMedQA (pqa_labeled, contexts given), Amazon ESCI (US, small version, test) and When2Call (MCQ test), run through deciding-with-confidence 0.4.0 with `eval --stack --transport cli` at low effort. `build.py` builds `items.jsonl`, `an.py` scores `stack.jsonl`; one-call figures use the plain-angle sample (`per_sample[0]`). Results are in the postscript of muninn.austegard.com/blog/haiku-answers-jev-requests.html.
