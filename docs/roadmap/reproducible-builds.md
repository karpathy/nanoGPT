# Reproducible Builds: Cache Debug Experiment

## Context

- GitHub Actions reported the repository at **52.79 GB** of cache usage vs a **10 GB** limit.
- Existing caches were pruned to prepare for a clean measurement of cache creation, eviction, and reuse semantics.

## Experiment Goals

- Observe first-run behavior after a full cache purge.
- Capture cache keys, sizes, and branch/PR associations after regeneration.
- Compare subsequent runs to confirm cache hits and identify optimization opportunities.

## Procedure

1. Enumerate caches: `gh cache list --json key,sizeInBytes,ref --limit 200 | tee /tmp/cache-inventory.json`.
1. Purge caches: `gh cache delete --all --succeed-on-no-caches`.
1. Create experiment branch: `git checkout -b chore/cache-debug && git push -u origin chore/cache-debug`.
1. Trigger workflow runs:
   - Push a marker commit to `chore/cache-debug` to start `quality.yml` (push trigger).
   - Optionally open a PR to exercise the `pull_request` trigger.
1. Monitor run: `gh run list --workflow quality.yml --limit 1` (record run ID), then `gh run watch <run-id> --exit-status`.
1. Collect regenerated caches: `gh cache list --json key,sizeInBytes,ref --limit 200 | tee /tmp/cache-postrun.json`.
1. Compare `cache-inventory.json` vs `cache-postrun.json` for growth and key patterns.

## Notes

- Empty commits (`git commit --allow-empty -m "chore: trigger cache debug"`) are acceptable to trigger push-based workflows.
- Delete the `chore/cache-debug` branch after completing measurements to avoid lingering cache keys.
- Retain the JSON inventories in `/tmp` only as long as needed; they are not stored in the repo.
