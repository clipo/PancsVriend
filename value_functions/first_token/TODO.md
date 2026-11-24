# Code to-dos: first-token vs path-sum comparison

Script: `first_token_vs_path_sum.py`. Tests: `test_first_token_vs_path_sum.py`.

- [ ] Run `--probe` on one record and fix the ADAPTER section if field names differ; add a test for the real layout
- [ ] Match `--eps` to the rule behind the census table's unsaturated counts (399 OLMo, 127 Qwen, 113 DeepSeek, ...)
- [ ] Tests tying the script to production: header settings equal `llm_runner` settings; `classify()` agrees with `MOVE_STAY_GRAMMAR`
- [ ] Full run on the 9 census models; save `results/e1_cells.csv` (not committed)
- [ ] Plot script for Fig. 2 (one panel per model, coloured by m_off, ±m_off band)
- [ ] E2: per-state mass by path type, from the stored trees
- [ ] E3: deviance G of FT-canon and FT-mask against the census counts, same bootstrap as the census test
- [ ] E4: grammar-conditional probability from the unmasked tables
- [ ] Generalise the classifier from MOVE/STAY to any finite option set (needed for X1 to X3)
