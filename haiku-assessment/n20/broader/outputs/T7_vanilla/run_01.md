# Task: Convert Natural Language Request to gh CLI Command

## Request
"show me the open PRs assigned to alice with the 'bug' label, sorted by oldest first"

## gh CLI Command

```bash
gh pr list --assignee alice --label bug --state open --sort created --order asc
```

## Explanation

- `gh pr list` - List pull requests
- `--assignee alice` - Filter to PRs assigned to the user 'alice'
- `--label bug` - Filter to PRs with the 'bug' label
- `--state open` - Show only open pull requests
- `--sort created` - Sort by creation date
- `--order asc` - Ascending order (oldest first)
