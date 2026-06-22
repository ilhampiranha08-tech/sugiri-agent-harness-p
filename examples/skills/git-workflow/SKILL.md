---
name: git-workflow
description: Git branching, committing, and PR workflow guidance. Use when the user needs help with git operations.
---

# Git Workflow Skill

## Common Operations

### Create a feature branch
```bash
git checkout -b feature/description
```

### Stage and commit changes
```bash
git add -A
git commit -m "type: description"
```

Commit types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### Interactive rebase
```bash
git rebase -i HEAD~N  # Last N commits
```

### Undo last commit (keep changes)
```bash
git reset --soft HEAD~1
```

### View history
```bash
git log --oneline --graph --all -20
```

## Best Practices

1. Commit early, commit often
2. Write meaningful commit messages
3. Keep PRs small and focused
4. Rebase before merging
5. Squash fixup commits before PR review
