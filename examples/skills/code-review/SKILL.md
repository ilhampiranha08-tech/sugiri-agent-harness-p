---
name: code-review
description: Review code for bugs, security issues, performance problems, and style issues. Use when the user asks for a code review.
---

# Code Review Skill

## Steps

1. Read the code file(s) the user wants reviewed
2. Check for:
   - **Bugs**: logic errors, off-by-one, null/undefined handling
   - **Security**: SQL injection, XSS, hardcoded secrets, unsafe eval
   - **Performance**: N+1 queries, unnecessary loops, memory leaks
   - **Style**: naming conventions, code organization, comments
3. Prioritize issues by severity (critical, high, medium, low)
4. Provide specific line references and suggested fixes
5. End with a summary of: files reviewed, issues found, overall quality score (1-10)

## Output Format

```
## Code Review: [filename]

### Critical Issues
- [Issue description with line number and fix]

### Warnings
- [Issue description with line number and fix]

### Suggestions
- [Improvement suggestion]

### Summary
- Files reviewed: N
- Issues found: N
- Quality score: X/10
```
