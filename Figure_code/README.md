# Benchmark calculations

Run `python Figure_code/run.py --profile all` from the repository root.
Use `--profile scRNA`, `--profile ATAC`, or `--profile Spatial` for one modality.
The root README documents inputs, formulas, missing sources, and output status.

The five dimension folders contain the calculations. `common/` owns shared
readers and numerical rules. `config/` declares each modality's input files,
method roster, metrics, and weights. `verify.py` independently recomputes scores
from Source_data. `tests/` covers scientific rules and missing-source behavior.

Existing figure scripts remain in `figures/`, with their own dependencies and
run commands. Core calculation does not require R or run plotting automatically.
`prepare_multimodal_inputs.py` documents optional extraction from the uploaded
archives and confirmed supplementary workbook. Prepared CSVs are distributed.
