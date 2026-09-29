# BRIEFING — 2026-09-29T12:46:00Z

## Mission
Execute tournament review tree [4, 2, 1] for Segment 6: paper02_experimental_evaluation_and_loso to conduct adversarial empirical review and produce unit_report_paper02_experimental_evaluation_and_loso.md.

## 🔒 My Identity
- Archetype: teamwork_preview_group
- Roles: orchestrator@document_review, successor
- Working directory: G:/S3_paper/.agents/teamwork/group_paper02_experimental_evaluation_and_loso
- Original parent: document_orchestrator_1
- Original parent conversation ID: 162469b4-bf2d-4563-9d8c-663bdc3dbf92

## 🔒 My Workflow
- **Pattern**: Document Review (Per-Segment RSA Tree [4, 2, 1], sample size 2)
- **Scope document**: G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md
1. **Decompose**: Segment 6 covers Sections 4, 5, 6, 7 of 03_research_paper_specimen_invariance/paper/main.tex and associated evaluation/trainer code.
2. **Dispatch & Execute**:
   - Level 0: 4 parallel Analysts (teamwork_preview_worker) producing 4 candidate reviews.
   - Level 1: 2 Review Aggregators (teamwork_preview_worker), each sampling 2 Level 0 candidates.
   - Level 2: 1 Final Aggregator (teamwork_preview_worker) synthesizing into unit_report_paper02_experimental_evaluation_and_loso.md.
3. **On failure**:
   - Retry: ping stuck agent
   - Replace: respawn fresh agent with partial progress
   - Skip / Degrade: only as last resort
4. **Succession**: Threshold at 16 spawns. Current plan uses 4 + 2 + 1 = 7 spawns.

## 🔒 Key Constraints
- NEVER produce analysis findings yourself — only dispatch, monitor, and aggregate via the tree protocol.
- Do NOT dispatch other group orchestrators or main orchestrators.
- Never reuse a subagent after it has delivered its handoff.
- Pass file paths, never inline report content.
- Final unit report must be written to G:/S3_paper/.agents/teamwork/group_paper02_experimental_evaluation_and_loso/unit_report_paper02_experimental_evaluation_and_loso.md.

## Current Parent
- Conversation ID: 162469b4-bf2d-4563-9d8c-663bdc3dbf92
- Updated: 2026-09-29T12:44:46Z

## Key Decisions Made
- Executing RSA review tree [4, 2, 1] with sample size 2.
- Analysts will audit SRI probe formulations, LOSO variance, 13 baseline benchmark parity, calibration ECE/MCE, and Grad-CAM forensic/legal alignment.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| Analyst 1 | teamwork_preview_worker | Level 0 Candidate 1 Review | in-progress | 840909d1-8382-4583-8b97-65c0644b6fb5 |
| Analyst 2 | teamwork_preview_worker | Level 0 Candidate 2 Review | in-progress | c9df2f25-75bf-455f-a920-8cb3fccaf9c3 |
| Analyst 3 | teamwork_preview_worker | Level 0 Candidate 3 Review | in-progress | 1efed869-abda-4276-81ba-cef2d2ca1ab8 |
| Analyst 4 | teamwork_preview_worker | Level 0 Candidate 4 Review | in-progress | f5e57509-8c17-4378-96f1-673eb659ee4e |

## Succession Status
- Succession required: no
- Spawn count: 4 / 16
- Pending subagents: 840909d1-8382-4583-8b97-65c0644b6fb5, c9df2f25-75bf-455f-a920-8cb3fccaf9c3, 1efed869-abda-4276-81ba-cef2d2ca1ab8, f5e57509-8c17-4378-96f1-673eb659ee4e
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: 5d6b48d5-c0ff-45f3-923d-6bbbda277516/task-36
- Safety timer: none

## Artifact Index
- G:/S3_paper/.agents/teamwork/document_orchestrator_1/ANALYSIS_PARTITION.md — Master analysis partition
- G:/S3_paper/.agents/teamwork/document_orchestrator_1/DOCUMENT_TEXT_MAP.md — Flattened document text map
- G:/S3_paper/03_research_paper_specimen_invariance/paper/main.tex — Target manuscript
