# LoCaLS 

**LoCaLS** learns the target-specific causal structure from observational data in the presence of **latent variables** and **selection bias**.

This repository accompanies the manuscript:

> *Local Causal Structure Learning in the Presence of Latent Variables and Selection Bias*

LoCaLS is designed for target-specific causal discovery tasks, where only the local causal structure around a given target variable is of interest.

## ✨ Features

This repository provides two versions of LoCaLS:

- **Data-driven LoCaLS** for observational data using Fisher's Z tests.
- **Oracle LoCaLS** for simulation studies using d-separation in a known DAG.

## ⚙️ Environment

The code was tested with:

- Python 3.11.15

| Package | Version |
| --- | ---: |
| NumPy | 1.26.4 |
| pandas | 2.1.4 |
| SciPy | 1.11.4 |
| NetworkX | 3.2.1 |
| igraph | 1.0.0 |
| causal-learn | 0.1.4.5 |
| pydot | 4.0.1 |

## 🚀 Quick Start

### Data-driven version

```python
import pandas as pd

from method.locaLS import data_LoCaLS

data = pd.read_csv("observations.csv")

result = data_LoCaLS(
    data,
    target="T",
    alpha=0.01,
    max_K=5,
)

pag = result["PAG.DataFrame"]
print(pag.loc["T"])
```

The input data should be finite, continuous, and numeric, with unique column names.

- `target`: target variable whose local causal structure is learned.
- `alpha`: significance level for conditional independence tests.
- `max_K`: maximum size of conditioning sets.

### Oracle version

For simulation studies with a known DAG, use:

```python
from method.locaLS import oracle_LoCaLS

result = oracle_LoCaLS(
    dag_adjacency,
    target="T",
    latent_nodes=["L1"],
    selection_bias_nodes=["S1"],
)
```

Here, `dag_adjacency` must be a square `pandas.DataFrame`.  
A nonzero entry at `[X, Y]` represents a directed edge `X -> Y`.

## 📦 Output

Both interfaces return a dictionary containing:

- `PAG.DataFrame`: the endpoint-mark adjacency matrix of the learned PAG.
- `CI_num`: number of conditional independence tests.
- `runtime_sec`: runtime in seconds.

Endpoint marks are encoded as:

| Value | Meaning |
| ---: | --- |
| 0 | no edge |
| 1 | circle |
| 2 | arrowhead |
| 3 | tail |

For an edge between `X` and `Y`:

- `[Y, X]` stores the endpoint mark at `X`;
- `[X, Y]` stores the endpoint mark at `Y`.

## 📊 Evaluation

Target-level evaluation can be performed using:

```python
from Utils.util_tools import local_mark_evaluation

metrics = local_mark_evaluation(
    pag,
    true_pag,
    target="T",
)
```

## 📓 Example Notebook

See [example.ipynb](example.ipynb) for a walkthrough of both the data-driven and oracle versions, including evaluation against the true PAG.

## 📚 Citation

```bibtex
@article{li2026locals,
  title  = {Local Causal Structure Learning in the Presence of Latent Variables and Selection Bias},
  year   = {2026}
}
```

## 📬 Contact

For questions or suggestions, please feel free to contact:

`zhengli0060(at)gmail(dot)com`

## 📄 License

This project is released under the [MIT License](LICENSE).
