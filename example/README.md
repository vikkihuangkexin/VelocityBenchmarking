### Sample file and simulated data

**Sample file contents**  
The example/sample file includes a simple table with the following columns:

- **data** — dataset index or short label  
- **ID** — dataset identifier  
- **Reference** — literature source or citation for the dataset  
- **Accession** — Accession ID or website

Example (illustrative):

| data | ID   | PMID           | Accession |
|------|------|----------------|-----------|
| 7_mouse_PancreaticE15.5_GSE132188 .h5ad    | 7    |    31160421    | GSE132188 |
| 24_mouse_brain.h5ad    | 24   |      NA        | https://www.10xgenomics.com/resources/datasets/fresh-embryonic-e-18-mouse-brain-5-k-1-standard-1-0-0  |
| 51_mouse_spermary_with_celltype.h5ad    | 51   |    37941145    |CNP0004694 |


**Simulated data generation**  
We generated 114 synthetic single-cell datasets with known ground-truth velocities using Dyngen (https://dyngen.dynverse.org/).  For each experiment, we instantiated both canonical developmental backbones and various backbones, including linear, bifurcating, bifurcating loop, consecutive bifurcating, trifurcating, cyclic and disconnect sturcture, sampled a gene-regulatory network and kinetic parameters (transcription, splicing, degradation) from Dyngen’s defaults, and simulated transcript dynamics via a stochastic reaction framework. Here, we only show an example for the code demo.
