import simplejson as json

if __name__ == "__main__":
    # References:
    # https://quantumcomputing.stackexchange.com/questions/34185/how-can-one-obtain-all-calibration-data-which-is-needed-to-plot-error-map-of-the
    # https://docs.quantum.ibm.com/api/migration-guides/local-simulators

    from datetime import datetime
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit.visualization import plot_error_map

    service = QiskitRuntimeService(
        channel="ibm_quantum",
        instance="ibm-q-hub-ntu/ntu-internal/default",
        token="ee4393fc53fe3dc29ff3765f60a46fd50eb650536a98580a688346c7bf275049012a3bb969d448aa6fced96fc62bd05a565a60d7fdfee7da80bbc01a434d5aa8",
    )
    backends = ["ibm_torino",
    # "ibm_sherbrooke",
    # "ibm_brisbane",
    # "ibm_osaka",
    # "ibm_nazca",
    # "ibm_kyoto",
    # "ibm_cusco",
    ]
    for backend_name in backends:
        backend = service.backend(name=backend_name)

        calibration_time = datetime(year=2024, month=5, day=21, hour=0, minute=0, second=0)
        properties = backend.properties(datetime=calibration_time)
        prop_dict = properties.to_dict()
        # print(prop_dict)
        with open(f"{backend_name}.json", "w") as outfile:
            json.dump(prop_dict, outfile, default=str)
        # Calculate EPLG

        # for i in range(50):
        #     print(1 - prop_dict["general"][-1 - i]["value"] ** (1 / (100 - 2 * i - 1)))
        # i = 44
        # print(1 - prop_dict["general"][-1 - i]["value"] ** (1 / (100 - 2 * i - 1)))
    # >>> print(prop_dict.keys())
    # dict_keys(['backend_name', 'backend_version', 'last_update_date', 'qubits', 'gates', 'general', 'general_qlists'])
    # >>> print(len(prop_dict['general']))
    # 338
    # >>> print(prop_dict['general'][-1]['name'])
    # lf_100 # My guess is layer fidelity
    # >>> print(prop_dict['general'][-1]['value'])
    # 0.15443334928335548
    # >>> lf_100_val = prop_dict['general'][-1]['value']
    # >>> 1 - lf_100_val ** (1/100)
    # 0.01850653819517256

    # print(prop_dict)
    # fake_backend = FakeBrisbane()
    # fake_backend._props_dict = properties.to_dict() # <== here
    # plot_error_map(fake_backend)
