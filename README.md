# QSD
Implementation of the quantum state discrimination in quantum circuits

## TACAS 2026 Artifact Submission
* The hyperlink to the artifact: []() (TBA).
* Additional requirements for the artifact: Install the required packages by executing the provided `./install.sh`.
* Detailed instruction: Please find the user manual below. Reviewers can see a quick demo by executing:
```bash
python quick_test_tacas.py
```
* Replication of results in the paper: Please see below for instructions.
Notice: The following flow is tested to be able to run on the `TACAS 2026 Artifact Evaluation Virtual Machine`. Please make sure that you are connected to the internet.

After you unzip the file, please enter the `SliQSim-master` directory and use the following command to ensure you have full permission to access files.

```commandline
chmod -R 777 ./*
```
To install, simply run:

```commandline
./install.sh
```
(TBA: Notebook filenames)
(TODO: Run the notebooks and check the time limits)
If not specified, the default time limit is 600 seconds as stated in the paper, which makes the whole flow take about 1 hour.
Feel free to set the time limit by yourself if you want to accelerate the process.

We apply for the "Functional", "Reusable", and "Available" badges.

Notice: The following is the user manual for general usage.

## Introduction
`QSD` is a software built for quantum state discrimination, that enables different strategies to find POVMs for QSD on top of `CVXPY` and provides conversion from these POVMs to executable quantum circuits.

## Download Source and Install Dependencies

### Download Source from Zenodo (TACAS 2026 artifact submission)
Link to the artifact: (TODO Change the url)
```bash
wget --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" https://figshare.com/ndownloader/files/49820907 -O QSD-TACAS2026.zip
chmod 777 QSD-TACAS2026.zip
unzip QSD-TACAS2026.zip
cd QSD
```

### Download Source from GitHub
```bash
git clone https://github.com/ChienKaiMa/QSD.git
cd QSD
git checkout work
```

### Install dependencies
`uv` is an extremely fast Python package and project manager, written in Rust. We will install most of the dependencies using `uv`.

This part is provided as a single file `install.sh`.
```bash
# You can execute the file without sudo
chmod 777 install.sh
./install.sh
```

```bash
# Make sure you're in the project folder
cd QSD

# Install uv if uv is not installed yet
wget -qO- https://astral.sh/uv/install.sh | sh

# Sync the dependencies in pyproject.toml
uv sync
# Activate the environment
source .venv/bin/activate
```
The optional dependencies are described in `optional_dep.md` and may provide acceleration for solving larger problems.

## Execution
All params that users can tweak (TBA)

(The three code blocks in the paper)
(TODO: Add more explanations)
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

