import os
import numpy as np
import matplotlib.pyplot as plt
from process_results import *


def exp0_plot(
    experiment_type="UQSD",
    num_points=40,
    num_rounds=4,
    shots=4096,
    n_qubit=2,
    p1=0.5,
    print_circ=True,
    sim=False,
    ibmq=True,
    backend="ibm_osaka",
    job_id="cqk9jybt65cg0087v79g",
):
    # TODO
    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS)

    fig = plt.figure(dpi=900)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)

    # x_axis = list(range(1, num_points)) * (1 / num_points)
    x_axis = np.array(range(1, num_points)) * (1 / num_points)
    print(x_axis)
    # Theoretical lines
    theo = []
    if experiment_type == "UQSD":
        theo = [1 - 0.5 * np.sqrt(i) for i in x_axis]
    elif experiment_type == "MED":
        theo = [0.5 * (1 + np.sqrt(1 - 4 * p1 * (1 - p1) * i)) for i in x_axis]

    plt.plot(x_axis, theo, "-", color=colors[0], alpha=0.5)

    from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Sampler
# 
    service = QiskitRuntimeService(channel="ibm_quantum")
    job = service.job(job_id)
    # service = BraketProvider().get_backend("Aria 1")
    # job = service.retrieve_job(job_id)
    print(job.status())


    # Assume the job is retrieved
    counts_list = job.result().get_counts()
    avg_hit_rate_list = []
    for i in range(1, num_points):
        # TODO Calculate the average atari rate
        atari_rate = 0
        for j in range(num_rounds):
            for key in counts_list[(i - 1) * num_rounds + j].keys():
                if experiment_type == "UQSD":
                    if key[0] == "0" and key[2:] == "00":
                        atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                    if key[0] == "1" and key[2:] == "10":
                        atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                    if key[2:] == "01":
                        atari_rate += 0.5 * counts_list[(i - 1) * num_rounds + j][key]
                    if key[2:] == "11":
                        atari_rate += 0.5 * counts_list[(i - 1) * num_rounds + j][key]
                if experiment_type == "MED":
                    if key[0] == "0" and key[2:] == "0":
                        atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                    if key[0] == "1" and key[2:] == "1":
                        atari_rate += counts_list[(i - 1) * num_rounds + j][key]

                print(key)
                pass
            pass
        atari_rate = atari_rate / shots / num_rounds
        avg_hit_rate_list.append(atari_rate)
    plt.plot(x_axis, avg_hit_rate_list, ".", label=f"p1 = {p1}", color=colors[0])

    plt.xlabel("c0 of Bob" + f" ({num_points} points) (n = {n_qubit})")
    plt.ylabel("Average hit rate")
    plt.legend(loc="lower right")
    plt.title(f"{experiment_type} ({backend})")
    plt.grid(True)
    plt.savefig(
        fname=f"{os.getcwd()}/exp0/results/"
        + f"exp0_result_{experiment_type}_{backend}"
        + f"_{p1}_{n_qubit}"
        + ".png",
        bbox_inches="tight",
    )
    plt.close()
    # TODO
    # Output counts to files
    return


exp0_plot(
    backend="ibm_osaka",
    job_id="cr08395k5z700081sd8g",
)

exp0_plot(
    backend="ibmq_kolkata",
    job_id="cr07yd1s9z7g008dr5v0",
)

# exp0_plot(
#     num_points=20,
#     num_rounds=1,
#     shots=500,
#     n_qubit=2,
#     p1=0.5,
#     backend="ionq_aria_1",
#     job_id="arn:aws:braket:us-east-1:513796107385:quantum-task/f77c9c41-2eb8-45ec-ac20-b747f389f973;arn:aws:braket:us-east-1:513796107385:quantum-task/ebed8768-b3eb-4746-b219-5d68b8ba0c5b;arn:aws:braket:us-east-1:513796107385:quantum-task/31f50ee2-3a3e-4936-ad36-384467c0dead;arn:aws:braket:us-east-1:513796107385:quantum-task/72e38659-983a-49d2-bf01-f179dd00534a;arn:aws:braket:us-east-1:513796107385:quantum-task/4f4b12b6-53f0-4781-a720-c64bab3f0e33;arn:aws:braket:us-east-1:513796107385:quantum-task/0823dca9-21ab-4cd8-8095-362aa6c85dfd;arn:aws:braket:us-east-1:513796107385:quantum-task/434130bb-c668-4274-9b72-441dcfd2b341;arn:aws:braket:us-east-1:513796107385:quantum-task/68cbd967-9a65-4948-ae40-1fa3327ddcf5;arn:aws:braket:us-east-1:513796107385:quantum-task/42c0d476-1d5f-433b-b173-6cbe733502e4;arn:aws:braket:us-east-1:513796107385:quantum-task/529ff6ab-b80b-4613-a647-055538878ba7;arn:aws:braket:us-east-1:513796107385:quantum-task/2db3b7be-6b12-4831-ab81-31aff4687b42;arn:aws:braket:us-east-1:513796107385:quantum-task/6a1b6f4b-efe0-4f7a-8de4-2cdd5d3241c0;arn:aws:braket:us-east-1:513796107385:quantum-task/93e1d3db-b9ef-476c-bc80-887cde61c587;arn:aws:braket:us-east-1:513796107385:quantum-task/718e52f5-3a43-4049-899d-94ce77a39989;arn:aws:braket:us-east-1:513796107385:quantum-task/785eed0b-2913-4db9-8388-604d60cbc484;arn:aws:braket:us-east-1:513796107385:quantum-task/c904ca4f-6d86-4a65-8a2d-2660f0b808ae;arn:aws:braket:us-east-1:513796107385:quantum-task/7172dd0c-b4ac-47e8-a81f-36b54558457f;arn:aws:braket:us-east-1:513796107385:quantum-task/c9de1789-e580-4a85-bf22-3607edf875cf;arn:aws:braket:us-east-1:513796107385:quantum-task/f0a3635a-a821-433e-b8f8-e35ea6bdc7dd",
#     ibmq=False,
# )
exp0_plot(
    experiment_type="MED",
    backend="ibmq_kolkata",
    job_id="cr0r7nedvs8g008j94ng",
)
