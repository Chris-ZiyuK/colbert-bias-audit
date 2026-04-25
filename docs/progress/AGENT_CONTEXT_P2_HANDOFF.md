# P2 Agent Handoff

Last updated: 2026-03-27
Repo: `colbert-bias-audit`
Scope: `P2` / `P2b` only
Audience: future LLM agents (`cursor`, `codex`, `antigravity`, etc.)
Primary goal: provide a reliable machine-readable context snapshot so the next agent can continue work without re-deriving the current state.

## 1. One-Screen Summary

### Current status

- `P2` is no longer framed as "prove a stable independent race main effect."
- The strongest stable signals remain:
  - `cross_gender`
  - `tokenization` under stricter matched designs
- `cross_race` is not stable in the old full-spec story, but in the new Rosenman-backed `P2b` targeted validation it becomes detectable again.
- `frequency_gap` is still not a stable independent effect.
- `absolute_rarity` is still the main unresolved hole.

### What changed in this turn

- The external source strategy was simplified:
  - `names-dataset` is no longer the preferred external source.
  - `Rosenman / Olivella / Imai 2023` is now the primary external first-name source for `P2b`.
- A new Rosenman-backed expansion matrix was built.
- A new targeted ColBERT validation was run on that matrix.
- The most important payoff:
  - `tokenization` moved from weak / underpowered to clearly significant.

### Most important headline result from this turn

From `results/p2_name_confound/p2b_rosenman_report.txt`:

- `tokenization`
  - `SS coef=+0.010138, p=0.002559`
  - `Func-TCD coef=+0.005980, p=0.001293`
- `gender`
  - `SS coef=+0.009331, p=0.0001535`
  - `Func-TCD coef=+0.005307, p=2.791e-05`
- `race`
  - `SS coef=+0.005051, p=0.04402`
  - `Func-TCD coef=+0.002827, p=0.03036`
- `frequency_gap`
  - not significant
- `absolute_rarity`
  - not significant
  - still severely underpowered

## 2. Claims Safe To Make

- `P2` supports a composite-effect interpretation rather than a single-factor race-only interpretation.
- `cross_gender` is the most stable name-side signal.
- `tokenization` now has materially stronger evidence than before, including significant Rosenman-backed targeted validation.
- `frequency_gap` does not currently survive as a stable independent effect in the stricter `P2b` designs.
- `cross_race` should be described carefully:
  - not a stable main effect in the original upgraded `P2` full-spec ladder
  - but detectable in the new Rosenman-backed targeted validation
  - best phrasing: "a residual identity-category signal remains detectable under the expanded targeted design"

## 3. Claims NOT Safe To Make

- Do NOT say `P2` has already proven a universally stable independent `race main effect`.
- Do NOT say `absolute rarity` has been cleanly identified.
- Do NOT say one external dataset has solved all name-side confounds.
- Do NOT treat Rosenman first-name probabilities as nationally perfect ground truth.
  - Caveat: they are based on six Southern-state voter files.

## 4. Canonical Files To Read First

Read these first if you need the shortest path to the current state:

- [`results/p2_name_confound/p2b_rosenman_report.txt`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_report.txt)
- [`results/p2_name_confound/p2b_rosenman_build_report.txt`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_build_report.txt)
- [`results/p2_name_confound/p2b_rosenman_gap_report.txt`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_gap_report.txt)
- [`data/audit_names/source_registry.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/source_registry.json)
- [`formal_findings.html`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/formal_findings.html)
- [`results/p2_name_confound/p2_final_report.txt`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2_final_report.txt)

## 5. External Source Decision

### Chosen source

- Primary external source for `P2b`:
  - `Rosenman, Olivella, Imai 2023`
  - Harvard Dataverse DOI: `10.7910/DVN/SGKW0K`

### Demoted source

- `names-dataset`
  - keep only as historical auxiliary validation
  - no longer the main external source for `P2b`

### Why this decision was made

- Rosenman is race-labeled and much more directly aligned with `P2b`'s expansion need.
- `names-dataset` was useful for race-agnostic tokenization / popularity checks, but it is not sufficient as the main source for race-labeled expansion.

### Important caveat

- Rosenman first-name probabilities are derived from six Southern-state voter files.
- This should be disclosed in any final paper text or external-facing summary.

## 6. New Files Added In This Turn

### New data artifacts

- [`data/audit_names/external/first_nameRaceProbs.tab`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/external/first_nameRaceProbs.tab)
- [`data/audit_names/external/first_raceNameProbs.tab`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/external/first_raceNameProbs.tab)
- [`data/audit_names/p2b_rosenman_additions.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/p2b_rosenman_additions.json)
- [`data/audit_names/p2b_rosenman_name_features.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/p2b_rosenman_name_features.json)
- [`data/audit_names/p2b_rosenman_matrix.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/p2b_rosenman_matrix.json)

### New scripts

- [`experiments/phase2_name_confound/build_p2b_rosenman_matrix.py`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/experiments/phase2_name_confound/build_p2b_rosenman_matrix.py)
- [`experiments/phase2_name_confound/p2b_rosenman_validation.py`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/experiments/phase2_name_confound/p2b_rosenman_validation.py)

### Updated scripts / metadata

- [`experiments/phase2_name_confound/p2b_gap_report.py`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/experiments/phase2_name_confound/p2b_gap_report.py)
- [`data/audit_names/source_registry.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/source_registry.json)

### New results

- [`results/p2_name_confound/p2b_rosenman_build_report.txt`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_build_report.txt)
- [`results/p2_name_confound/p2b_rosenman_gap_report.txt`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_gap_report.txt)
- [`results/p2_name_confound/p2b_rosenman_gap_summary.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_gap_summary.json)
- [`results/p2_name_confound/p2b_rosenman_rows.csv`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_rows.csv)
- [`results/p2_name_confound/p2b_rosenman_summary.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_summary.json)
- [`results/p2_name_confound/p2b_rosenman_report.txt`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_report.txt)
- [`results/p2_name_confound/p2b_rosenman_effects.png`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_effects.png)

## 7. Exact Quantitative Change From Old Verified Matrix To New Rosenman Matrix

### Pair-count change

Old matrix: [`data/audit_names/p2b_verified_matrix.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/p2b_verified_matrix.json)
New matrix: [`data/audit_names/p2b_rosenman_matrix.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/p2b_rosenman_matrix.json)

- `race`
  - old: `35` total pairs, `22` treatment, `13` control
  - new: `59` total pairs, `40` treatment, `19` control
- `gender`
  - old: `27` total pairs, `14` treatment, `13` control
  - new: `44` total pairs, `25` treatment, `19` control
- `tokenization`
  - old: `17` total pairs, `4` treatment, `13` control
  - new: `49` total pairs, `30` treatment, `19` control
- `frequency_gap`
  - old: `35` total pairs, `22` treatment, `13` control
  - new: `43` total pairs, `24` treatment, `19` control
- `absolute_rarity`
  - old: `4` total pairs, `2` treatment, `2` control
  - new: `5` total pairs, `2` treatment, `3` control

### Interpretation of the change

- The critical improvement is in `tokenization`.
- `absolute_rarity` remains underpowered.
- Therefore:
  - the new source integration materially improved one major unresolved mechanism
  - it did NOT fully close the last hole

## 8. Current Best Interpretation Of P2

Use this internal summary:

`P2` measures a composite effect made of profession context, tokenization / language-processability, gender contrast, and a weaker residual race-category signal. The upgraded Rosenman-backed `P2b` materially strengthens the tokenization story and makes the race residual more detectable under the expanded targeted design, but absolute rarity remains unresolved.

## 9. Remaining Bottleneck Inside P2

### Main unresolved item

- `absolute_rarity`

### Why it is unresolved

- Even after Rosenman expansion:
  - `absolute_rarity` has only `5` pairs
  - result is not significant

### Practical implication

- `P2` is now much stronger than before.
- But if the next agent wants to "finish P2," the best target is:
  - add more names specifically to increase `absolute_rarity` coverage
  - not to broadly increase everything again

## 10. Next Best Actions

Priority order for the next agent:

1. Improve `absolute_rarity` coverage using the existing Rosenman-backed workflow.
2. Update narrative docs to reflect the new Rosenman-backed evidence.
3. Only after `P2` text is synchronized, move back to whole-paper gaps:
   - real-text generalization
   - multi-model comparison

### Concrete next action if continuing P2 immediately

- Add another conservative batch of Rosenman names that specifically target:
  - same `race`
  - same `gender`
  - same tokenizer count
  - near-equal frequency
  - broader spread in mean rarity

## 11. Rerun Commands

Run from repo root.

### Build Rosenman matrix

```bash
conda run -n colbert_bias python experiments/phase2_name_confound/build_p2b_rosenman_matrix.py
```

### Build Rosenman gap report

```bash
P2B_VERIFIED_PATH=/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/p2b_rosenman_name_features.json \
P2B_MATRIX_PATH=/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/p2b_rosenman_matrix.json \
P2B_GAP_REPORT_PATH=/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_gap_report.txt \
P2B_GAP_SUMMARY_PATH=/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/results/p2_name_confound/p2b_rosenman_gap_summary.json \
P2B_GAP_TITLE='P2b Rosenman gap report' \
conda run -n colbert_bias python experiments/phase2_name_confound/p2b_gap_report.py
```

### Run targeted validation

```bash
conda run -n colbert_bias python experiments/phase2_name_confound/p2b_rosenman_validation.py
```

## 12. Important Implementation Notes

- The Rosenman table itself does NOT include `gender`.
- To avoid introducing another external database, this turn used a conservative manually gender-curated additions file:
  - [`data/audit_names/p2b_rosenman_additions.json`](/Users/ziyukong/codebase/BRN/CS2952W/Proposal/colbert-bias-audit/data/audit_names/p2b_rosenman_additions.json)
- This means:
  - the current Rosenman-backed expansion is intentionally conservative
  - future expansion can continue using the same pattern without changing the methodological story

### Frequency handling

- The Rosenman-backed matrix uses `P(name | race)` prevalence scaled to per-100k within assigned race as the main frequency value for matching.
- This is stored as:
  - `rosenman_prevalence_per_100k`
  - `freq_value_for_matching`
  - `log_freq`

### Missing item

- `Thiruvengadam` was not found in Rosenman tables and remains a fallback anchor.

## 13. Whole-Paper Reality Check

If the next agent is asked "is this publishable now?", answer carefully:

- `P2` specifically:
  - close to paper-worthy
  - much stronger than before
  - not fully closed because `absolute_rarity` is still weak
- full project:
  - still missing at least:
    - real-text generalization
    - multi-model contrast

## 14. Minimal Decision Policy For Future Agents

If uncertain, follow this policy:

- Do not add a second new external first-name database unless the user explicitly changes direction.
- Stay with the Rosenman source as the main external source.
- Treat `absolute_rarity` as the only major unresolved `P2b` mechanism issue.
- Keep all public-facing claims conservative.
- Prefer improving evidence quality over multiplying datasets.

