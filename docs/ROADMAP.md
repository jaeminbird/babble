# Roadmap: from demo to serious prototype

babble today: one six-router FRR network, 13 scenarios, Opus 5.5 solves 65/65 at k = 5. That result is real but saturated: the bench can no longer tell whether a change to the harness helps. The next phase makes the problem harder (large, multi-vendor networks with realistic faults) and the harness stronger, using mechanisms that have published evidence behind them.

Every phase keeps the design principles in `CLAUDE.md`. A tool or dependency comes in only when a measured problem needs it.

## Phase 0: finish the demo

| Item | Status |
|---|---|
| Fail-fast fixes, renames, dead-code removal, CI, mypy | Done |
| A trap scenario whose obvious fix breaks something | Done: `shared-prefix-list-trap`. Opus avoided it 5/5; `tests/test_live_trap.py` shows the rollback on live routers |
| k = 5 trials per scenario | Done: 65/65, pass^5 = 1.00, $0.075 per trial (2026-10-25) |

## Phase 1: an eval that doesn't saturate

Nothing below can be shown to help without this, so it comes first.

- **False-premise tickets.** FaulT-Bench ([arXiv 2608.27021](https://arxiv.org/abs/2608.27021)) found frontier agents saturate on accurate tickets but degrade sharply when the network is healthy and the ticket is wrong. Grow the two control scenarios into four classes: wrong device named, wrong cause suggested, healthy network, vague ticket, each written in several reporter styles. Score over-diagnosis.
- **Generated scenarios.** Fault templates (inject, verify, ground truth) applied across topologies, so the bench grows by combination instead of hand-written YAML.
- **Grade the end state, not the path**, with pass^k at k >= 5 ([Anthropic, evals for agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

## Phase 2: harness mechanisms with evidence

| Mechanism | Evidence | In babble |
|---|---|---|
| Phase gate: no fix before a committed hypothesis | SADE ([arXiv 2605.04530](https://arxiv.org/abs/2605.04530)): +22 points root-cause F1 from the policy alone, same model | `propose_change` is unavailable until `commit_hypothesis(device, category, evidence_ids)` cites earlier tool results. Enforced by the harness, not the prompt. |
| Hypothesis ledger outside the conversation | [Long-running harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents), [Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) | JSON ledger (untested / ruled in / ruled out + evidence) that survives compaction; replaces free-form notes |
| Independent verifier before approval | SWE-PRM ([arXiv 2509.02360](https://arxiv.org/pdf/2509.02360)): 40.0% to 50.6% on SWE-bench Verified | A fresh-context subagent asks "does the evidence support this, and what alternative wasn't ruled out?" Its answer is shown to the approver. |
| Explicit abstention tools | AgentAbstain ([arXiv 2607.10059](https://arxiv.org/abs/2607.10059)) | `declare_no_fault` and `escalate_physical`, each requiring health evidence |
| Higher-level reads and actionable errors | [Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | `bgp_session_health(scope)`, `path_trace(source, destination)` next to raw `show` |
| Advisor and server-side context management as ablation arms | [Advisor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool), context editing and compaction | Presets measured on cost per solved task |

## Phase 3: multiple vendors through typed changes

Raw config lines don't scale across vendors: every vendor would need its own destructive-line denylist. ANI-Gamut ([arXiv 2609.22723](https://arxiv.org/abs/2609.22723)) found typed transactions passed 6/6 where raw shell passed 0/6, at about a tenth of the cost.

- **Device adapters behind one interface**: normalized reads (`get_bgp_neighbors`, `get_routes`, `get_interfaces`) and typed writes (`bgp_neighbor.set(remote_as=...)`, `interface.set(enabled=...)`), rendered per vendor.
  - FRR: vtysh JSON (what babble does today).
  - Nokia SR Linux: JSON-RPC over HTTP, with its native candidate datastore and confirmed commit.
  - Third vendor: Juniper cJunosEvolved via NETCONF (decided 2026-10-25). Free and needs no account; it runs under KVM, so the first task is proving it boots inside WSL.
- **The agent's tool schema stays identical across vendors.** The policy layer, approval gate and audit log don't change.
- **Native confirmed commit where the device has it**, emulated snapshot-and-restore where it doesn't (FRR).

## Phase 4: scale

- **Topologies from netlab** ([netlab.tools](https://netlab.tools/platforms/)): one topology file generates addressing and configs across FRR, SR Linux, cEOS and Junos, and `netlab validate` provides service checks.
- **Target: about 80 to 100 nodes on this machine**, mostly FRR, plus about 6 SR Linux and 2 to 3 nodes of the third vendor. SR Linux costs about 1.2 GB per node; FRR is about 20 MB.
- **Fault classes small labs miss**: route leaks, route-reflector misconfiguration, MTU/fragmentation over tunnels, one bad ECMP member, BFD mismatch, VRF and route-target leaks, EVPN/VXLAN faults, and cross-vendor policy defaults (SR Linux and Junos reject eBGP routes by default; FRR doesn't).
- **Context at scale**: programmatic tool calling for fan-out ("BGP summary on all 100 routers, return non-Established peers") and deferred tool loading per vendor ([advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)).

## Phase 5: only when a measured need appears

| Candidate | When it earns its place |
|---|---|
| Batfish pre-change check | When the bench shows bad changes that live verification catches too late (it doesn't parse SR Linux) |
| MCP server exposing the gated tools | When another harness should use babble's tools; the approval gate stays inside babble |
| Trust ladder (auto-approve per action class, earned by pass^k) | Once pass^k numbers exist per action class |
| Kubernetes via Clabernetes | Only for multi-host labs or parallel bench runs; still 0.x |
| Terraform | Only to provision a larger cloud KVM host or seed NetBox; there is no containerlab provider |
| NetBox | When drift from the source of truth becomes a fault class |

## Deliberately not doing

- **Multi-agent debate**: doesn't reliably beat self-consistency at equal compute ([arXiv 2502.08788](https://arxiv.org/abs/2502.08788)).
- **Tree search on a live network**: changes are stateful; search belongs in a simulation, not on routers.
- **A bare-shell agent**: works for code, loses for networks (ANI-Gamut).
- **Cross-incident memory**: stale lessons from one topology are a correctness risk.

## Decisions made

1. **Third vendor**: Juniper cJunosEvolved (2026-10-25), over Arista cEOS.
2. **Budget for bench runs**: $10 for Phase 0 (2026-10-25). A Phase 1 sweep on a generated bench needs its own budget.
