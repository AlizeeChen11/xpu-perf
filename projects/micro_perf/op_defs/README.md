## op_defs Maintenance Guide

This directory defines operator metadata used by micro_perf. It is maintained as a standalone sub-module with its own version and change history.

### Scope

- Owns operator definition Python modules under `basic_ops/` and `llm_ops/`.
- Owns operator-level documentation for maintainers and users.
- Does not own backend kernels or runtime execution logic.

### Directory Layout

- `VERSION`: semantic version for this sub-module.
- `CHANGELOG.md`: release notes for each op_defs version.
- `basic_ops/`: common operator definitions.
- `llm_ops/`: llm-oriented operator definitions.
- `llm_ops/llm_ops.md`: detailed llm operator specification.

### Versioning Policy

`op_defs` uses semantic versioning: `MAJOR.MINOR.PATCH`.

- MAJOR: incompatible change in op schema/behavior.
- MINOR: backward-compatible feature (new op, new optional args, new arg type).
- PATCH: backward-compatible fix (docs fix, bug fix without schema break).

### Change Recording

Update `CHANGELOG.md` whenever operator definitions, arguments, or behavior change.

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
4. If the change is user-visible, update docs (for llm ops, update `llm_ops/llm_ops.md`).
5. Before release, move `[Unreleased]` entries into a new version section with date.

### Definition Change Checklist

- Operator name stable and clear.
- Input/output shapes and dtypes documented.
- Attr list complete and default values explicit.
- Arg type behavior consistent with existing ops.
- Provider implementation impact assessed.

### Compatibility Notes

- Renaming/removing attrs is a breaking change.
- Reinterpreting existing attr semantics is a breaking change.
- Adding optional attrs with safe defaults is non-breaking.
