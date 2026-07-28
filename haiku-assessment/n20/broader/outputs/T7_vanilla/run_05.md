# GH CLI Command for PR Query

## Request
Show me the open PRs assigned to alice with the 'bug' label, sorted by oldest first

## Command
```bash
gh pr list --assignee alice --label bug --state open --sort created --order asc
```

## Breakdown
- `gh pr list` - List pull requests
- `--assignee alice` - Filter to PRs assigned to alice
- `--label bug` - Filter to PRs with the 'bug' label
- `--state open` - Show only open PRs
- `--sort created` - Sort by creation date
- `--order asc` - Order ascending (oldest first)
