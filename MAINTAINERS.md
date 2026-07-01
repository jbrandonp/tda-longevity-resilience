# Maintainers

## Current

| Name | GitHub | Email | Role |
|------|--------|-------|------|
| Brandon Palhano Machado | [@jbrandonp](https://github.com/jbrandonp) | brandonpalhano@gmail.com | Lead Developer |

## Maintainer Responsibilities

- Review and merge pull requests
- Triage issues and feature requests
- Manage releases (version bump, CHANGELOG, tag, PyPI)
- Keep dependencies up to date (Dependabot)
- Ensure CI/CD pipelines remain green
- Respond to security reports

## Release Process

1. Update version in `setup.py` and `src/__init__.py`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag vX.Y.Z`
4. Push tag: `git push --tags`
5. Create GitHub Release with release notes
6. Update Zenodo DOI
7. (Future) Push to PyPI and conda-forge

## Joining

Interested in contributing? Start with issues tagged `good first issue`.
Consistent contributors may be invited as maintainers.
