# gh CLI Command: Open PRs Assigned to Alice with Bug Label

## Request
Show me the open PRs assigned to alice with the 'bug' label, sorted by oldest first

## Command
```bash
gh pr list --assignee alice --label bug --state open --sort created --order asc
```

## Explanation
- `gh pr list` - Lists pull requests
- `--assignee alice` - Filters to PRs assigned to the user 'alice'
- `--label bug` - Filters to PRs with the 'bug' label
- `--state open` - Shows only open PRs
- `--sort created` - Sorts by creation date
- `--order asc` - Orders in ascending order (oldest first)
