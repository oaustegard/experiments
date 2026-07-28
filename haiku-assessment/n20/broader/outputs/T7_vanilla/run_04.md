# GitHub CLI Command for PR Query

## Request
Show me the open PRs assigned to alice with the 'bug' label, sorted by oldest first

## gh CLI Command

```bash
gh pr list --assignee alice --label bug --state open --sort created --order asc
```

## Command Breakdown

- `gh pr list` — List pull requests
- `--assignee alice` — Filter to PRs assigned to alice
- `--label bug` — Filter to PRs with the 'bug' label
- `--state open` — Show only open PRs
- `--sort created` — Sort by creation date
- `--order asc` — Oldest first (ascending order)

## Alternative Syntax

The same command using short flags:

```bash
gh pr list -a alice -l bug -s open --sort created --order asc
```
