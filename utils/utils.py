def get_case_id(nq, ns, seed):
    return f"q{nq}_n{ns}_s{seed}"


def get_qasm_name_by_case(case_id):
    return f"qc_iso_{case_id}_no_backend.qasm"
