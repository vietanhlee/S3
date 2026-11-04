# BRIEFING — 2026-11-04T04:02:00Z

## Mission
Orchestrate a team of up to 3 agents to refactor and complete the research paper in `03_research_paper_specimen_invariance/paper/main.tex` and execute supplementary experiments in `03_research_paper_specimen_invariance/` under the scientific reframing "Specimen Leakage Audit and Honest Negative Result Reporting for Invariance Methods".

## 🔒 My Identity
- Archetype: orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: g:/S3_paper/.agents/teamwork/orchestrator
- Original parent: sentinel (cadbab1c-f46f-47c9-9b4a-01e020b746c0)
- Original parent conversation ID: cadbab1c-f46f-47c9-9b4a-01e020b746c0

## 🔒 My Workflow
- **Pattern**: Project Orchestration (team cap: 3 agents)
- **Scope document**: g:/S3_paper/.agents/teamwork/orchestrator/PROJECT.md
1. **Decompose**:
   - Agent 1 (Explorer): Deep technical survey of `03_research_paper_specimen_invariance/` code, data, seeds, and `paper/main.tex`. Map all 8 contradictions and experiment scripts needed.
   - Agent 2 (Worker): Run supplementary experiments (9 seeds random-split CI, specimen 50 & 53 diagnosis, GRL discriminator logging), rewrite `paper/main.tex` to implement R1, R2, R3, and compile clean `main.pdf`.
   - Agent 3 (Reviewer/Auditor): Verify compilation, audit against all 8 contradiction resolutions, check tables/numbers/species names, ensure zero overclaiming, and perform integrity checks.
2. **Dispatch & Execute**:
   - Sequential specialized agents respecting max 3 agents constraint.
3. **On failure**:
   - Retry / Replace within budget.
4. **Succession**:
   - Self-succeed if spawns reach threshold (here hard-capped at 3).

## 🔒 Key Constraints
- Maximum 3 agents total.
- DISPATCH-ONLY: Do not write code or edit LaTeX directly in target repo; delegate to subagents.
- Never write, modify, or create source code directly.
- All implementations must be genuine production quality.
- Zero tolerance for integrity violations.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.

## Current Parent
- Conversation ID: cadbab1c-f46f-47c9-9b4a-01e020b746c0
- Updated: 2026-11-04T04:02:00Z

## Key Decisions Made
- Allocated 3-agent pipeline: Explorer -> Worker -> Reviewer/Auditor to strictly comply with max 3 agents limit while maintaining full research integrity and verification.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_m1 | teamwork_explorer | Deep codebase survey, 9 seeds data, specimen 50/53, LaTeX contradictions | completed | 5f10d310-902c-43e0-9b2d-23ac8c645bd8 |
| worker_m2 | teamwork_worker | Code experiments, paper refactoring, LaTeX compilation | completed | d9566922-4eb8-469a-801b-948a2d78d01f |
| auditor_m3 | teamwork_auditor | Independent audit of paper, experiments, and integrity verification | completed | 7df75995-c709-4dc3-9cd5-eef7a3bdf5db |

## Succession Status
- Succession required: no
- Spawn count: 3 / 3
- Pending subagents: none
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: killed
- Safety timer: none

## Artifact Index
- g:/S3_paper/.agents/teamwork/ORIGINAL_REQUEST.md — Original user request
- g:/S3_paper/.agents/teamwork/orchestrator/DISPATCH.md — Dispatch log
- g:/S3_paper/.agents/teamwork/orchestrator/BRIEFING.md — This briefing
- g:/S3_paper/.agents/teamwork/orchestrator/progress.md — Progress tracker
- g:/S3_paper/PROJECT.md — Global project scope and architecture
