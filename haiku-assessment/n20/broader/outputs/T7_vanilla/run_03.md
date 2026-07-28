# gh CLI Command Conversion

## Request
"show me the open PRs assigned to alice with the 'bug' label, sorted by oldest first"

## gh CLI Command
```bash
gh pr list --assignee alice --label bug --state open --sort created --order asc
```

## Explanation
- `gh pr list` — list pull requests
- `--assignee alice` — filter to PRs assigned to alice
- `--label bug` — filter to PRs with the 'bug' label
- `--state open` — filter to open PRs only
- `--sort created` — sort by creation date
- `--order asc` — ascending order (oldest first)
