# Venues and deadlines

Last checked: 18 Nov 2026. Sources are listed under each entry.

## Current plan

- **Scope this round:** MOVE vs STAY only, because that is the data we have.
- **Target:** ACL 2027 through ARR, deadline **4 Jan 2027 (real, AoE)**.
- **Open:** short (4 pp) or long (8 pp). Decide after E1 (see the gate below).
- **Must-do for the paper:** other choice types (multi-option, shared-prefix labels, integers). See `todo_experiments.md`. If they can't be ready by 4 Jan, they go in the extended version.

## Venues considered

| Venue | Format | Dates | Fit | Status |
|---|---|---|---|---|
| **ACL 2027** (Kyoto, 17 to 22 Aug 2027) | Short 4 pp or long 8 pp, plus 1 page at camera-ready | ARR deadline 4 Jan 2027 (real, AoE) | Good. Readout methods for LLM answers are an ACL topic, and the closest prior work is ACL-family | **Target** |
| ICML 2027 (Rio de Janeiro, 31 Jul to 6 Aug 2027) | 8 pp | Abstract 14 Jan 2027, paper 22 Jan 2027 (real, AoE); author list frozen at the abstract deadline | Weak this round. Needs about 4 new tasks and a backend check (about 27 working days) | Not this round |
| TMLR | Rolling, 10 to 12 pp typical | No deadline | Strong. Judged on whether claims are supported, not on novelty size | Fallback, and home for the extended version |
| COLM 2027 | Full paper | Not announced; 2024 to 2026 deadlines were 29 to 31 Mar (unverified for 2027) | Good for the extended version | Watch |
| NeurIPS 2027 (main or Datasets and Benchmarks) | Full paper | Not announced; a tracker projects mid to late May (unverified) | Possible for the extended version | Watch |
| EMNLP 2027 via ARR | Short or long | No 2027 cycle listed yet beyond January | Possible resubmission route | Watch |

Sources:
- ACL 2027 call: https://2027.aclweb.org/calls/main/
- ICML 2027 call: https://icml.cc/Conferences/2027/CallForPapers
- COLM past deadlines: https://mldeadlines.com/conference/colm/
- ARR dates: https://aclrollingreview.org/dates

## Decision gate (after E1)

Run E1 (first-token vs path sum) by **5 Dec 2026 (assigned)**.

- **If m_off or |path - FT-mask| exceeds 0.01 on at least 20% of unsaturated states in 2 or more models:** the path sum corrects the shortcut. Submit to ACL on 4 Jan.
- **If not:** the path sum is a certificate for MOVE vs STAY. The paper then needs a task where the first token does not decide the answer (shared-prefix labels, `todo_experiments.md` X1) before it is strong. Either add a small version of X1 by 20 Dec (assigned) and submit short, or skip 4 Jan and send the extended version to TMLR or COLM.

## Other dates that affect the schedule

- 10 Dec 2026: CBR Pizza Seminar talk (DCT paper, first priority).
- End of Dec 2026: move to the US.
- Authorship and overlap with the Schelling paper must be agreed with Andreas Pape before submission.
