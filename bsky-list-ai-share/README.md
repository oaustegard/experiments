# Which list members post least about AI?

Oskar asked which members of his Bluesky list "Artificial (and real) Intelligence" post too little about AI
to stay on it. This measures each member's share of posts about AI over the 100,000-post snapshot behind
austegard.com/bsky/answer-search (2026-09-08 to 2026-11-04).

**Method.** 600 random posts were labelled about-AI or not by a headless Claude Haiku (`label.sh`, keyed
JSON replies: bare arrays came back 101-108 items long). A logistic regression on the posts' Ettin-17m
vectors (`score.py`, C=3) reaches cross-validated AUC 0.885 and is calibrated (mean predicted 0.323 against
0.328 labelled; `results/model.json`), so a member's share is the mean predicted probability over their posts,
with a bootstrap interval over those posts.

**Result.** 310 of 564 members posted in the window (254 posted nothing). Shares run continuously from about
0.1 to over 0.6, median 0.37, with no natural gap. 29 members with at least 30 posts have an upper 95% bound
below 0.25; together they wrote 22% of the corpus and 27% of its non-AI posts. The per-member table
(`data/members.json`) is not committed because it ranks named people.

**Limits.** The labeller counts a short reply as not-AI unless its topic shows, so members who reply a lot in
AI threads score low; the classifier is English-centred and undercounts posts in other languages; 600 labels
are one labeller's judgement, not a gold set.

## Finding members to add (`discover.py`, `vet.py`)

Authenticated post search (the public AppView refuses `searchPosts`) for 40 AI terms over 14 days gave
10,281 posts from 5,277 non-members; 587 appeared at least 3 times and 570 had enough posts to score with the
same classifier. The top of that ranking is bots, feeds and marketers, so `vet.py` screens (AI share >= 0.40,
<= 40 posts a day, no bot or feed words), re-fetches 100 posts for reply share and samples, and has a headless
Haiku judge whether each is an individual writing about AI (`vet_judge.sh`). 207 passed the screen, 41 the
judge; they went on a review list for Oskar. Names stay in `data/`.

## Quieter candidates from the follow graph (`follows.py`, then `vet.py`)

Search only finds people who used an exact term lately. `follows.py` reads every member's follows (550
members, 105,730 non-members followed, 275 s), takes non-members followed by at least 12 members (3,732)
minus the candidates list and the accounts removed on 2026-11-04, scores the 500 most-followed (462 had
enough posts), and `vet.py data/follow_candidates.json data/follow_vetted.json` screens them (AI share
>= 0.40, <= 40 posts a day, a post in the last 30 days, no bot or feed words) and has Haiku judge the 66
left: 48 individuals, followed by 43 to 127 members each. A Bridgy mirror of a member's Mastodon account
was dropped by hand, and 47 went on a second review list. The judge rejects organizations (conference,
lab and company accounts), which a member list may want; they are in `data/follow_vetted.json`.
