# QSD

Implementation of the quantum state discrimination in quantum circuits.

## TACAS 2026 Artifact Submission

* **Artifact link:** []() *(TBA)*
* We apply for the "Functional", "Reusable", and "Available" badges.
* The artifact has been verified to run on the **TACAS 2026 Artifact Evaluation Virtual Machine** with Intel/AMD 64-bit host architecture (`TACAS26-AEC-amd64.ova`).
  Please ensure that your environment has an active internet connection for installing third-party dependencies.

### Installation

After unzipping the artifact, navigate to the `QSD-master` directory and ensure you have full access permissions by using:

```bash
chmod -R 777 ./*
```

Then, install the dependencies and set up the environment by executing:

```bash
source ./install.sh
```

### Smoke Test

For a quick functionality check, reviewers can run a quick demo using:

```bash
python3 quick_test_tacas.py
```

### Replication of Results in the Paper

To reproduce all experimental results presented in the paper, execute the following command:

```bash
python3 all.py
```

The entire process is expected to take about **??? hours**.
All generated results will be stored in the `results` directory.

* `results/crossqsd_results.csv`: raw data for **Fig. 1**, visualized as `results/crossqsd_ratio.png`
* `results/fitqsd_results.csv`: raw data for **Fig. 2**, visualized as `results/fitqsd.png`
* `???.csv`: raw data for **Fig. 3**, visualized as `???.png`
* `???.csv`: raw data for **Fig. 4**, visualized as `???.png`
* `???.csv`: raw data for **Table 3**, summarized in `???.csv`
* `???.csv`: raw data for **Fig. 7**, visualized as `???.png`

---

> **Notice:**   
> The following sections are the general user manual beyond the TACAS evaluation scope.

## Introduction

`QSD` is a software framework for quantum state discrimination that supports various strategies for finding POVMs using `CVXPY` and enables the automated conversion of these POVMs into executable quantum circuits.


## Download Source and Install Dependencies

### Download from GitHub

Clone the repository ~~and switch to the working branch~~ by:

```bash
git clone https://github.com/ChienKaiMa/QSD.git
cd QSD
```

### Install Dependencies

Most dependencies can be installed automatically using `uv`, a fast Python package and project manager written in Rust.
To set up the environment, run:

```bash
chmod 777 install.sh
source install.sh
```

Optional dependencies for performance enhancement are listed in `optional_dep.md`, which may accelerate large-scale problem solving.


## Execution (TODO: Add more explanations)

### ProblemSpec: Problem instance construction
```python
from utils.handy_states import simple_2 
from flow.solve_mix import *

state_dict = simple_2(0.2, 0.5, 0.7) # predefined quantum state set 
num_qubits = state_dict["num_qubits"]
num_states = state_dict["num_states"]
dense_mat = state_dict["dense_mat"]
qsd_problem = ProblemSpec(
    num_qubits=num_qubits, 
    num_states=num_states, 
    case_id="test" # name tag
)

qsd_problem.prior_prob = np.ones(num_states) * (1 / num_states) 
qsd_problem.set_states(
    state_type="densitymatrix", 
    states=dense_mat, 
    overwrite=True,
)
```

### POVM synthesis with an optimization strategy
```python
from flow.build_circuits import *

crossqsd_prob = apply_crossQSD(
    qsd_problem,
    cvxpy_settings={"solver": cp.SCS, "verbose": False}, 
    alpha=[0.02] * qsd_problem.num_states,
    beta=[0.02] * qsd_problem.num_states
)

povm_vectors = get_povm_vectors(crossqsd_prob)
povm_ckt = POVMCircuit(povm_vectors=np.stack(povm_vectors)) 
povm_ckt.fix()
```
### (Optional) Quantum circuit synthesis and resynthesis
```python
from flow.resynth import resynth_aqc 
qc = povm_ckt.build_circuit("ccd")
qc = resynth_aqc(qc)
```

