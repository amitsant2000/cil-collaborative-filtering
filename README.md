# Collaborative Filtering — CIL 2025  
**Authors:** Amit Sant, Anuj Srivastava, Florentin Aigner, Shubh Goel

## Overview  
This repository contains the implementation of our collaborative filtering approach developed for the CIL 2025 project. The core model logic is implemented in `two_towers.py`, and the `Experiments` directory contains training logs from our experiments.

## Running the Code  
To run our method on the student cluster, use the following command:

```bash
bash run.sh [SEED] [USE_NCE]
```
### Arguments
* [SEED]: Random seed for reproducibility (e.g., 1337)
* [USE_NCE]: Set to 1 to enable contrastive learning (NCE), 2 to enable CL with multiple threshold, or 0 to disable it

### Example
`bash run.sh 1337 1`
