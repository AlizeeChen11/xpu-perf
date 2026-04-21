## vendor_ops Maintenance Guide

This directory contains vendor-specific operator implementations for micro_perf. It is maintained as a standalone sub-module with its own semantic version and changelog.

### Scope

- Owns vendor implementations under each hardware directory (for example `GPU/`, `NPU/`, `MLU/`).
- Owns provider registration and vendor-side operator compatibility.
- Does not own base operator schema (that belongs to `op_defs/`).

### Directory Layout

- `VERSION`: semantic version for this sub-module.
- `CHANGELOG.md`: release notes for vendor implementations.
- `<VENDOR>/env.json`: runtime environment requirements.
- `<VENDOR>/ops/`: vendor operator implementations.
- `<VENDOR>/projects/`: vendor tools and profiling projects.

### Versioning Policy

`vendor_ops` uses semantic versioning: `MAJOR.MINOR.PATCH`.

- MAJOR: incompatible provider behavior change or removed provider/op support.
- MINOR: backward-compatible provider additions (new vendor op/provider support).
- PATCH: backward-compatible fixes (bug fix, docs fix, non-breaking perf fix).

### Change Recording

Update `CHANGELOG.md` whenever vendor implementation behavior changes.

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
4. If op behavior or constraints changed, sync related docs in vendor `README.md`.
5. Before release, move `[Unreleased]` entries into a dated version section.

### Compatibility Checklist

- Provider registration name unchanged (unless major bump).
- Supported dtype/shape/attr constraints documented.
- Degradation path is clear if optional dependency is missing.
- Vendor-specific limitations are explicitly documented.

### Relation With op_defs

- `op_defs` defines operator schema and benchmark arguments.
- `vendor_ops` provides vendor-side implementations and provider metadata.
- If `op_defs` has a breaking schema change, evaluate coordinated major bump in `vendor_ops`.
