# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | ✅ Active support  |
| < 1.0   | ❌ Pre-release     |

## Reporting a Vulnerability

**Do NOT open a public issue.** Send vulnerability reports to:
- Email: brandonpalhano@gmail.com
- Subject: `[SECURITY] tda-longevity-resilience — <summary>`

You will receive an acknowledgment within 48 hours and a status update within 7 days.

## Security Practices

### What we do
- **Dependabot** scans dependencies weekly for known CVEs
- **Bandit** static analysis runs on every commit
- No execution of untrusted pickle files (diagram cache uses hash-verified `.pkl`)
- No API keys or secrets in the repository

### What you should do
- Keep your environment up to date (`conda update --all`)
- Verify pickle cache files before loading from untrusted sources
- Do not commit data files containing personally identifiable information (PII)

## Known Risks (Accepted)

| Risk | Mitigation |
|------|-----------|
| Pickle deserialization in diagram cache | Cache files are hash-verified; only load from `data/processed/` |
| `ripser` native code execution | Official PyPI wheels only; verify checksums |
| User-uploaded CSV in Streamlit dashboard | pandas `read_csv` with default parsing; no `eval` or `exec` |

## Security Tools

```bash
# Run bandit scan
bandit -c .bandit -r src/

# Check dependencies for known vulnerabilities
pip-audit  # or: safety check

# Pre-commit hooks (auto-run)
pre-commit run --all-files
```
