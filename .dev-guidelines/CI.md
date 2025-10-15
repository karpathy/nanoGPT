---
trigger: manual
description: Operational standards for GitHub Actions workflows and CI maintenance
---

# Continuous Integration Guidelines

Operational policies for authoring, maintaining, and monitoring CI workflows in this repository.

<details>
<summary>Related documentation</summary>

- [Development Practices](./DEVELOPMENT.md) – Core development practices, quality standards, and workflow for ml_playground contributors.
- [Testing Standards](./TESTING.md) – Exhaustive testing policies enforcing TDD, coverage, and determinism.
- [Documentation Guidelines](./DOCUMENTATION.md) – Standards for structure, abstraction, and formatting across documentation.
- [.github README](../.github/README.md) – Current GitHub Actions implementation details and workflow inventory.

</details>

## Table of Contents

- [Guiding Principles](#guiding-principles)
- [Workflow Layout](#workflow-layout)
- [Caching Strategy](#caching-strategy)
- [Execution Tiers](#execution-tiers)
- [Monitoring and Reruns](#monitoring-and-reruns)
- [Maintenance Checklist](#maintenance-checklist)

## Guiding Principles

- **Parity with local workflow.** Ensure CI invokes the same Typer CLIs (`ci-tasks`, `env-tasks`, `test-tasks`) that developers run locally.
- **Fast feedback first.** Maintain a short-running gate workflow that exercises linting, typing, and smoke tests before any long-running suites.
- **Current tooling.** Track upstream releases of the automation platform, actions, and runtime toolchains. Upgrade promptly to retain security fixes and performance improvements.
- **Transparent control flow.** Avoid implicit skips or silent fallbacks—document conditions and surface logs when a step is bypassed.

## Workflow Layout

- Define shared environment variables and defaults at the job level so every step inherits a consistent context.
- Prefer a single job per workflow unless parallel fan-out measurably cuts wall time. Document the rationale for multi-job workflows inline.
- Configure explicit timeouts that reflect realistic upper bounds for each job to prevent runaway compute consumption.
- Provide manual triggers for investigative or long-running suites so they can be dispatched on demand without impacting the primary gate.

## Caching Strategy

- Cache compiled dependencies (virtual environments, wheels, etc.) using keys derived from immutable inputs such as lock files, runtime versions, and operating system.
- Keep caches scoped to a single concern (virtual environment, lint caches, build artifacts) to avoid unnecessary invalidations and oversized uploads.
- Prune caches as part of the workflow when they are mutated to prevent uncontrolled growth, while leaving restored environments intact.
- Capture platform-specific prerequisites (system packages, toolchain dependencies) in reusable setup steps so cache misses remain deterministic.
- Refer to `../.github/README.md` for the concrete cache keys and caching actions currently in use.

## Execution Tiers

- Maintain at least two workflow tiers:
  - **Gate workflows** run on every push/pull request with minimal runtime and exhaustive failure visibility.
  - **Extended workflows** run on schedules or manual dispatch for mutation testing, benchmarks, or integration suites.
- Document SLA, trigger strategy, and expected runtime for each tier in the workflow repository README.
- Ensure extended workflows publish artifacts and metrics that diagnose regressions without requiring live log access.

## Monitoring and Reruns

- Use the automation platform’s streaming log tools to monitor in-flight jobs without waiting for completion.
- Cancel superseded or stalled executions promptly to free runners for high-priority work.
- Prefer rerunning individual jobs or workflows targeting the specific revision under investigation, ensuring the latest workflow definition is applied.
- Record manual interventions (reruns, cancellations, timeouts) in the associated PR or issue for reviewer context.

## Maintenance Checklist

- **On tooling upgrades:** Validate changes in a feature branch, confirm cache behavior, and sync the `.github` documentation with any new requirements.
- **When workflows change:** Update this guideline with revised policies and refresh the `.github` README with the concrete implementation.
- **Quarterly review:** Confirm scheduled workflows remain necessary, secrets are rotated on time, and runtime budgets still align with expectations.
- **Post-incident follow-up:** Capture remediation steps for recurring failures or cache anomalies to shorten future triage cycles.
