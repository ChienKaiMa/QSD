from utils.handy_states import simple_2
from flow.solve_mix import *

print("Start the smoke test.")
state_dict = simple_2(0.2, 0.5, 0.7)  # predefined quantum state set
num_qubits = state_dict["num_qubits"]
num_states = state_dict["num_states"]
dense_mat = state_dict["dense_mat"]
qsd_problem = ProblemSpec(
    num_qubits=num_qubits, num_states=num_states, case_id="test"  # name tag
)
qsd_problem.prior_prob = np.ones(num_states) * (1 / num_states)
qsd_problem.set_states(
    state_type="densitymatrix",
    states=dense_mat,
    overwrite=True,
)
print("Successfully initialized a QSD ProblemSpec.")


from flow.build_circuits import *

crossqsd_prob = apply_crossQSD(
    qsd_problem,
    cvxpy_settings={"solver": cp.SCS, "verbose": False},
    alpha=[0.02] * qsd_problem.num_states,
    beta=[0.02] * qsd_problem.num_states,
)
print("Successfully found a POVM solution.")

povm_vectors = get_povm_vectors(crossqsd_prob)
povm_ckt = POVMCircuit(povm_vectors=np.stack(povm_vectors))
povm_ckt.fix()
povm_ckt.build_circuit()
print("Successfully built a quantum circuit for QSD!")
print("Smoke test passed!")
