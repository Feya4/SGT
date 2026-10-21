# Code that produced the reported runs

The runs reported in the manuscript (`sgt200_bracs*` and its controls) were trained with the earlier
implementation in `sgt_impl13/`. Its training and model code is archived here unchanged, so that every reported
number can be traced to the exact code that produced it.

The current code (repository root) reproduces these runs within seed variation:
SGT re-trained with it reaches 48.75 ± 1.96% test weighted F1 on BRACS, against 47.77 ± 1.74% for the original
runs (seeds 0–4; `run_controls.py --controls sgt_rerun`).

<!-- Copy here, unchanged: sgt_impl13/train*.py, model*.py, dataset*.py and any other .py files used for training. -->
