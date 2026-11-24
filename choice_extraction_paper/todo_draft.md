# Draft to-dos

Draft lives in `draft/` (LaTeX). Dates are **assigned**.

## Now
- [ ] Port to the official ACL template: copy `acl.sty` and `acl_natbib.bst` from https://github.com/acl-org/acl-style-files into `draft/` (main.tex switches to them automatically)
- [ ] Fix the opening of the status note: draft 2, not draft 1

## After E1 (5 Dec)
- [ ] §4 headline sentence and numbers; abstract sentence on first-token deviation
- [ ] Decide framing: correction (large m_off) or certificate (small m_off); rewrite intro paragraph 3 and contribution 4 to match
- [ ] Fig. 2: FT-canon and FT-mask vs path sum, one panel per model, coloured by m_off, with the ±m_off band

## Figures and tables (18 Dec)
- [ ] Fig. 1 prefix-tree schematic for one state
- [ ] Fig. 3 path sum vs census MOVE rate, with binomial intervals; residual histogram
- [ ] Fig. 4 error budget per model
- [ ] Table 1 update with re-census G and p for all 9 models
- [ ] Table 2 (E2) or one sentence
- [ ] Table 3 numbers checked against the batch-numerics and infrastructure notes

## Text
- [ ] Confirm from the path-sum code that B is accumulated relative to the allowed mass at each state
- [ ] Report the total mass on 6 to 8 token paths (agents stop at 5 tokens)
- [ ] Compare text-level and token-level prefixes on a few states
- [ ] Define "unsaturated" exactly as the census code does
- [ ] Fold related work into §1 and §2 if going short; move Table 3, the Qwen item and the checklist to the appendix
- [ ] References: full author lists, venues; Schwager et al. initials; authors of 2607.17948

## Before submission (22 to 28 Dec)
- [ ] Anonymise; appendix with grammar, payloads, census CSV
- [ ] Limitations and Responsible NLP checklist
- [ ] Co-author read-through
