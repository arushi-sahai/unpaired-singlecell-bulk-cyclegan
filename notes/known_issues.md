# Known issues

A few stale references remain in code that is left unedited. The training runs are cached under keys that hash `dgm/translation.py`, so editing it would retrain every cached model, and `dgm/testeval.py` is kept exactly as it was audited for the test run.
- `dgm/translation.py` (lines 3 and 579) and `dgm/testeval.py` (line 141) call the analysis plan "prereg", and the first points to `notes/phase3_prereg.md`. The plan is now `notes/phase3_plan.md`. It was an internal plan fixed before training, not a public registration.
- The test frontier curve drawn on the results page was traced from Phase 7's saved figure, because Phase 7 saved the margins but not the curve as a table (`results/phase8/test_frontier_traced.csv`, `notes/deviations.md` row 23). The trace matches the saved frontier value at the CycleGAN's accuracy, and every margin comes from the saved computation.
- `notes/phase3_plan.md` is kept as written. It says the test split is scored "in Phase 6", and the test evaluation later became Phase 7.
