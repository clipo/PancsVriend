# Paper checklist

Everything the paper needs before submission. Status: [ ] open, [~] in progress, [x] done.
Dates are **assigned** unless marked real.

## Decisions
- [x] Scope this round: MOVE vs STAY only (18 Nov 2026)
- [x] Target: ACL 2027 via ARR, 4 Jan 2027 (real)
- [ ] Short or long: decide after E1 (5 Dec)
- [ ] Authorship, and which paper reports the census: agree with Andreas Pape (25 Nov)
- [ ] ARR rules on preprints and overlapping work under review: check (1 Dec)

## Evidence (details in `todo_experiments.md`)
- [~] E1 first token vs path sum (5 Dec)
- [ ] E0 finish re-census (Mistral, DeepSeek), Qwen-cell adjudication, all 9 G values (1 Dec)
- [ ] E2 mass by path type (8 Dec)
- [ ] E3 deviance test per readout (10 Dec)
- [ ] E4 grammar-conditional vs sampler law (10 Dec)
- [ ] Other choice types (X1 to X3): must-do for the paper; in the January submission only if the gate requires it, otherwise in the extended version
- [ ] Every number in Tables 1 and 3 checked against the repo

## Draft (details in `todo_draft.md`)
- [~] LaTeX draft builds (`draft/main.tex`)
- [ ] Figures 1 to 4
- [ ] All `\tbd{}` marks filled
- [ ] Fits the page limit (4 or 8 pp) in the ACL template
- [ ] References complete (full author lists, venues)
- [ ] Limitations section (required by ACL)
- [ ] Responsible NLP checklist (required by ARR)
- [ ] Anonymised: no names, no repo links, self-citations in third person

## Release
- [ ] Code for the extractor and the comparison, with tests, cleaned for release
- [ ] Census CSV and per-state tables in the supplementary material
- [ ] Co-author read-through (28 Dec)
- [ ] Submit via OpenReview (4 Jan 2027, real, AoE)
