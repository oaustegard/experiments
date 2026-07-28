#!/bin/bash
# For each spoke, list every non-default branch with:
#   - whether it's merged into default
#   - last commit date
#   - associated PR number (if any)
#   - commits ahead of main

SPOKE_REPOS="remex claude-skills claude-workspace oaustegard.github.io blog-references browser-extensions bookmarklets aeyu.io muninn.austegard.com claude-container-layers eml-sr remax muninns-inbox jina-v5-nano-mirror muninn-utilities claude-tangled-spoke claude-jjithub-and-spoke claude-github-and-spoke claude-workspace-fuse remax_kb install-manifest-spec mojo-bm25s fusemojo tree-sitter-mojo thinking-traces-eval llm-as-computer transformer-vm lemur-numpy Lottie_Playlist container-layer-test muninn-backup"

OUT=/tmp/branch-audit.tsv
echo -e "repo\tbranch\tdefault\tahead\tbehind\tmerged_via_pr\topen_pr\tlast_commit_age" > "$OUT"

for repo in $SPOKE_REPOS; do
  # Get default branch
  default=$(gh api "repos/oaustegard/$repo" --jq '.default_branch' 2>/dev/null)
  if [ -z "$default" ]; then continue; fi

  # Get all branches with their commit SHA
  branches=$(gh api "repos/oaustegard/$repo/branches?per_page=100" --jq '.[] | "\(.name)\t\(.commit.sha)"' 2>/dev/null)

  # Get open PR head branches
  open_prs=$(gh pr list --repo "oaustegard/$repo" --state open --json number,headRefName --limit 100 2>/dev/null | python3 -c "import sys,json; [print(p['headRefName']) for p in json.load(sys.stdin)]")

  # Get all PRs (open + closed) keyed by head ref → number/state
  all_prs=$(gh pr list --repo "oaustegard/$repo" --state all --json number,headRefName,state,mergedAt --limit 200 2>/dev/null)

  echo "$branches" | while IFS=$'\t' read -r branch sha; do
    [ "$branch" = "$default" ] && continue
    [ -z "$branch" ] && continue

    # Open PR?
    open_pr=$(echo "$open_prs" | grep -Fx "$branch" | head -1)

    # PR state for this branch
    pr_info=$(echo "$all_prs" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for p in data:
    if p['headRefName'] == '$branch':
        print(f\"{p['number']}|{p['state']}|{p['mergedAt'] or ''}\")
        break
" 2>/dev/null)

    # Compare branch vs default
    cmp=$(gh api "repos/oaustegard/$repo/compare/$default...$branch" --jq '{ahead:.ahead_by, behind:.behind_by}' 2>/dev/null)
    ahead=$(echo "$cmp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ahead',''))" 2>/dev/null)
    behind=$(echo "$cmp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('behind',''))" 2>/dev/null)

    # Last commit date
    last_date=$(gh api "repos/oaustegard/$repo/commits/$sha" --jq '.commit.committer.date' 2>/dev/null)

    echo -e "$repo\t$branch\t$default\t$ahead\t$behind\t$pr_info\t$open_pr\t$last_date" >> "$OUT"
  done
done

echo "Done. Results in $OUT"
wc -l "$OUT"
