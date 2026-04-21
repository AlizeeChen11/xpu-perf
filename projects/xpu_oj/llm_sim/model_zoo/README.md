## llm_sim/model_zoo Maintenance Guide

This directory contains model topology templates and model configuration mappings used by `projects/xpu_oj/llm_sim`.

It is maintained as a standalone sub-module with its own semantic version and changelog.

### Scope

- Owns model family mappings and topology composition logic.
- Owns deploy templates and source/model config metadata under model family directories.
- Does not own runtime benchmark engine logic in `llm_sim/endpoint.py`.

### Directory Layout

- `VERSION`: semantic version for this sub-module.
- `CHANGELOG.md`: release notes for model_zoo changes.
- `topology.py`, `op_templates.py`: topology and operator template definitions.
- `<model_family>/`: model configs, source metadata, deploy templates.

### Versioning Policy

`model_zoo` uses semantic versioning: `MAJOR.MINOR.PATCH`.

- MAJOR: incompatible topology/config schema changes.
- MINOR: backward-compatible model family or deploy template additions.
- PATCH: backward-compatible fixes (docs/config corrections, non-breaking bug fixes).

### Change Recording

Update `CHANGELOG.md` whenever model topology/configuration behavior changes.

Use these sections:

- Added
- Changed
- Deprecated
- Removed
- Fixed
- Breaking

### Release Process

When a PR changes files in this directory:

1. Decide version bump level (major/minor/patch).
2. Update `VERSION`.
3. Add an entry under `[Unreleased]` in `CHANGELOG.md`.
4. If behavior is user-visible, update docs and template comments.
5. Before release, move `[Unreleased]` entries into a dated version section.

### Compatibility Checklist

- `BASE_MODEL_MAPPING` keys remain stable unless major bump.
- Topology generation contract remains backward-compatible unless major bump.
- Required config fields are documented and validated.
- Deploy template behavior changes are documented.
