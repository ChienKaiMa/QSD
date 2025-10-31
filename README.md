# QSD

Implementation of the quantum state discrimination in quantum circuits.

## TACAS 2026 Artifact Submission

* **Artifact link:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17493285.svg)](https://doi.org/10.5281/zenodo.17493285)
* We apply for the **"Functional"**, **"Reusable"**, and **"Available"** badges.
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

The entire process is expected to take several hours to complete.  
As the replication of **Table 3** accounts for most of the runtime, users may include the `--qubits {n_qubit}` parameter, where `n_qubit` $\in \{2, 3, 4, 5\}$, to generate results only for rows up to `n_qubit` in **Table 3**.  
All generated results will be stored in the `results` directory.  
* `crossqsd_results.csv`: raw data for **Fig. 1**, visualized as `crossqsd_ratio.png`
* `fitqsd_results.csv`: raw data for **Fig. 2**, visualized as `fitqsd.png`
* `hybrid_psucc_results_3states.csv`: raw data for **Fig. 3 (left)**, visualized as `hybrid_3states_psucc.png`
* `hybrid_sqrtd_results_3states.csv`: raw data for **Fig. 3 (right)**, visualized as `hybrid_3states_sqrtd.png`
* `hybrid_psucc_results_2states.csv`: raw data for **Fig. 4 (left)**, visualized as `hybrid_2states_psucc.png`
* `hybrid_sqrtd_results_2states.csv`: raw data for **Fig. 4 (right)**, visualized as `hybrid_2states_sqrtd.png`
* `qc_opt_summary_table.csv`: raw data for **Table 3**, summarized in `qc_opt_summary_table.csv`
* `qc_sim.csv`: raw data for **Fig. 7**, visualized as `OptUQSD_ccd_q6_n3.png`

---

> **Notice:**   
> The following sections are the general user manual beyond the TACAS evaluation scope.

## Introduction

`QSD` is a software framework for quantum state discrimination that supports various strategies for finding POVMs using `CVXPY` and enables the automated conversion of these POVMs into executable quantum circuits.


## Download Source and Install Dependencies

### Download from GitHub

Clone and enter the repository by:

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


## Execution

### ProblemSpec: Problem instance construction
Use the class `ProblemSpec` to start a quantum state discrimination (QSD) instance.
The `ProblemSpec` object records the quantum states generally in numpy arrays, either
as a state vector or density matrix and other basic information such as the number of qubits
used to describe the states `num_qubits` and the number of states to be discriminated `num_states`.
Users can provide a tag `case_id` to better track the objects later.

User can specify any quantum states they like, as long as the states are expressed
in iterable arrays. `ProblemSpec` also provides a `set_states` function so that
users can specify or change the states' data later to reuse the objects.

Below is an example:
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
We can then construct various optimization problems using functions in `flow/solve_mix.py`.

| function name      | state type     | QSD          | extra arguments        |
| ------------------ | -------------- | ------------ | ---------------------- |
| apply_Eldar        | state vector   | Optimal UQSD | beta                   |
| med_problem        | density matrix | MED          | (None)                 |
| med_plus_problem   | density matrix | MED+         | (None)                 |
| apply_frio         | density matrix | FRIO         | p_inc_lb               |
| apply_crossQSD     | density matrix | CrossQSD     | alpha, beta            |
| min_l1_problem     | density matrix | FitQSD-minL1 | ideal_distrib          |
| min_ss_problem     | density matrix | FitQSD-minSS | ideal_distrib          |
| meco_problem       | density matrix | FitQSD-MECO  | ideal_distrib          |
| hybrid_obj_problem | density matrix | hybrid obj.  | ideal_distrib, param_a |

The below code example continues the previous code block. The code solves a POVM according to the CrossQSD strategy, and extract the solution POVM with `get_povm_vectors`. We can then construct a `POVMCircuit` instance before the quantum circuit actually gets synthesized. `fix` makes sure that the input array is a POVM
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
`flow.build_circuits` has the function to build a quantum circuit from a POVM using isometry synthesis. First we can synthesize the circuit with `ccd`, `csd`, or `knill` methods. There are various functions to optimize a given quantum circuit in `flow.resynth`, for example `resynth_unitary`, `resynth_unitary_approx`, and `resynth_aqc`.
The following code example follows the previous code block and obtains a resynthesized quantum circuit for `CrossQSD`.
```python
from flow.resynth import resynth_aqc 
qc = povm_ckt.build_circuit("ccd")
qc = resynth_aqc(qc)
```

