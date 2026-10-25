# faultline

**Network incidents take too long to diagnose, and operators won't let AI touch production networks to help.**

Models can already reason about router configs. What's missing is trust: nobody lets an agent run commands on a live router unless they know exactly what it can do, can see what it did, and can undo it. That makes it a harness problem more than a model problem.

faultline is a Claude agent harness built from the raw Messages API around that constraint, plus an eval bench that measures two things:

1. **Effectiveness:** does it find the root cause of a network fault, how fast, and at what cost?
2. **Safety:** zero unapproved changes, every change verified, automatic rollback when a fix fails, and a full audit trail.

## How it works

1. An alert comes in ("site B can't reach the data center").
2. The agent investigates read-only. It can hand each router to a subagent that inspects it with fresh context and reports back a summary.
3. It names the root cause, with evidence.
4. It proposes a fix. Config changes need approval.
5. It applies the fix, then the harness verifies it (reachability, BGP state) and rolls back automatically if the checks fail.
6. Every command, decision and result goes to an audit log.

The practice network is [containerlab](https://containerlab.dev) running FRRouting containers. The bench breaks it in realistic ways and checks the fix automatically.

## Status

Under construction. See `docs/PLAN.md`.
