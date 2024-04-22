import os
import numpy as np
import matplotlib.pyplot as plt

# from qiskit_braket_provider import *
from process_results import *


def plot_theo(experiment_type, p1, color, num_points=40):
    theo_x_axis = np.array(range(1, num_points)) * (1 / num_points)
    theo = []
    if experiment_type == "UQSD":
        theo = [1 - 0.5 * np.sqrt(i) for i in theo_x_axis]
    elif experiment_type == "MED":
        theo = [0.5 * (1 + np.sqrt(1 - 4 * p1 * (1 - p1) * i)) for i in theo_x_axis]
    theo_sqrt_x_axis = [np.sqrt(i) for i in theo_x_axis]
    plt.plot(theo_sqrt_x_axis, theo, "-", color=color, alpha=0.5)


def plot_ibmq(
    experiment_type,
    num_points,
    num_rounds,
    shots,
    p1,
    color,
    job_id,
):
    from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Sampler

    service = QiskitRuntimeService(channel="ibm_quantum")
    job = service.job(job_id)
    # service = BraketProvider().get_backend("Aria 1")
    # job = service.retrieve_job(job_id)
    print(job.status())
    counts_list = job.result().get_counts()
    avg_hit_rate_list = []
    x_axis = np.array(range(1, num_points)) * (1 / num_points)
    sqrt_x_axis = [np.sqrt(i) for i in x_axis]
    if num_rounds * num_points > len(counts_list):
        num_rounds = 1
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
                    if key[0] == "0" and key[2] == "0":
                        atari_rate += counts_list[(i - 1) * num_rounds + j][key]
                    if key[0] == "1" and key[2] == "1":
                        atari_rate += counts_list[(i - 1) * num_rounds + j][key]

                print(key, counts_list[(i - 1) * num_rounds + j][key])
                pass
            pass
        atari_rate = atari_rate / shots / num_rounds
        avg_hit_rate_list.append(atari_rate)
    plt.plot(
        sqrt_x_axis,
        avg_hit_rate_list,
        ".",
        label=f"p1 = {p1}, {job.backend().name}",
        color=color,
    )


def plot_ionq(experiment_type, num_points, num_rounds, shots, p1, color):
    # Load results from file
    counts_list = load_aria1_results(experiment_type=experiment_type)
    # counts_list = job.result().get_counts()
    x_axis = np.array(range(1, num_points)) * (1 / num_points)
    sqrt_x_axis = [np.sqrt(i) for i in x_axis]
    avg_hit_rate_list = []
    for i in range(1, num_points):
        # TODO Calculate the average atari rate
        atari_rate = 0
        for j in range(num_rounds):
            index = (i - 1) * num_rounds + j
            for key in counts_list[index].keys():
                if experiment_type == "UQSD":
                    if key[-1] == "0" and key[1:3] == "00":
                        print(key, key[-1], key[1:3])
                        atari_rate += counts_list[index][key]
                    if key[-1] == "1" and key[1:3] == "01":
                        atari_rate += counts_list[index][key]
                    if key[1:3] == "10":
                        atari_rate += 0.5 * counts_list[index][key]
                    if key[1:3] == "11":
                        atari_rate += 0.5 * counts_list[index][key]
                if experiment_type == "MED":
                    if key[-1] == "0" and key[-2] == "0":
                        atari_rate += counts_list[index][key]
                    if key[-1] == "1" and key[-2] == "1":
                        atari_rate += counts_list[index][key]

                print(key, counts_list[index][key])
                pass
            pass
        atari_rate = atari_rate / shots / num_rounds
        avg_hit_rate_list.append(atari_rate)
    plt.plot(sqrt_x_axis, avg_hit_rate_list, ".", label=f"p1 = {p1}, ionq", color=color)


def exp0_plot(
    experiment_type="UQSD",
    num_points=20,
    num_rounds=1,
    shots=500,
    n_qubit=2,
    p1=0.5,
    # num_points=40,
    # num_rounds=4,
    # shots=4096,
    print_circ=True,
    sim=False,
    ibmq=True,
    ionq=False,
    backend="ibm_osaka",
    job_id_list=["cqk9jybt65cg0087v79g"],
):
    """Print theoretical line, and IBMQ, IonQ results."""
    # TODO
    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS)

    fig = plt.figure(dpi=300)
    fig.set_figwidth(6)
    fig.set_figheight(4.8)

    # Theoretical lines
    plot_theo(experiment_type, p1, colors[0], 40)

    # Assume the job is retrieved
    color_id = 1
    # round_list = [1, 4, 4]
    for job_id in job_id_list:
        plot_ibmq(
            experiment_type,
            num_points=40,
            num_rounds=4,
            shots=4096,
            p1=p1,
            color=colors[color_id],
            job_id=job_id,
        )
        color_id += 1

    plot_ionq(
        experiment_type,
        num_points=20,
        num_rounds=1,
        shots=500,
        p1=0.5,
        color=colors[color_id],
    )

    # plt.xlabel("c0 of Bob" + f" ({num_points} points) (n = {n_qubit})")
    plt.xlabel(r"$\sqrt{c0}$ of Bob" + f" (n = {n_qubit})")
    plt.ylabel("Average hit rate")
    plt.legend(loc="upper right")
    plt.title(f"{experiment_type}")
    plt.grid(True)
    plt.savefig(
        fname=f"{os.getcwd()}/exp0/results/"
        + f"exp0_result_{experiment_type}_all"
        + f"_{p1}_{n_qubit}"
        + ".png",
        bbox_inches="tight",
    )
    plt.close()
    # TODO
    # Output counts to files
    return


exp0_plot(
    experiment_type="UQSD",
    job_id_list=[
        "cr08395k5z700081sd8g",
        "cr07yd1s9z7g008dr5v0",
    ],
)

exp0_plot(
    experiment_type="MED",
    job_id_list=[
        "co07874madoqp48gflp0",
        "cr0r7nedvs8g008j94ng",
        "cr0sapts9z7g008dsq2g",
    ],
)

# exp0_plot(
#     experiment_type="UQSD",
#     backend="ibm_osaka",
#     job_id="cr08395k5z700081sd8g",
# )
# exp0_plot(
#     experiment_type="UQSD",
#     backend="ibmq_kolkata",
#     job_id="cr07yd1s9z7g008dr5v0",
# )
# print(
#     """
# exp0_plot(
#     num_points=20,
#     num_rounds=1,
#     shots=500,
#     n_qubit=2,
#     p1=0.5,
#     backend="ionq_aria_1",
#     job_id="arn:aws:braket:us-east-1:513796107385:quantum-task/f77c9c41-2eb8-45ec-ac20-b747f389f973;arn:aws:braket:us-east-1:513796107385:quantum-task/ebed8768-b3eb-4746-b219-5d68b8ba0c5b;arn:aws:braket:us-east-1:513796107385:quantum-task/31f50ee2-3a3e-4936-ad36-384467c0dead;arn:aws:braket:us-east-1:513796107385:quantum-task/72e38659-983a-49d2-bf01-f179dd00534a;arn:aws:braket:us-east-1:513796107385:quantum-task/4f4b12b6-53f0-4781-a720-c64bab3f0e33;arn:aws:braket:us-east-1:513796107385:quantum-task/0823dca9-21ab-4cd8-8095-362aa6c85dfd;arn:aws:braket:us-east-1:513796107385:quantum-task/434130bb-c668-4274-9b72-441dcfd2b341;arn:aws:braket:us-east-1:513796107385:quantum-task/68cbd967-9a65-4948-ae40-1fa3327ddcf5;arn:aws:braket:us-east-1:513796107385:quantum-task/42c0d476-1d5f-433b-b173-6cbe733502e4;arn:aws:braket:us-east-1:513796107385:quantum-task/529ff6ab-b80b-4613-a647-055538878ba7;arn:aws:braket:us-east-1:513796107385:quantum-task/2db3b7be-6b12-4831-ab81-31aff4687b42;arn:aws:braket:us-east-1:513796107385:quantum-task/6a1b6f4b-efe0-4f7a-8de4-2cdd5d3241c0;arn:aws:braket:us-east-1:513796107385:quantum-task/93e1d3db-b9ef-476c-bc80-887cde61c587;arn:aws:braket:us-east-1:513796107385:quantum-task/718e52f5-3a43-4049-899d-94ce77a39989;arn:aws:braket:us-east-1:513796107385:quantum-task/785eed0b-2913-4db9-8388-604d60cbc484;arn:aws:braket:us-east-1:513796107385:quantum-task/c904ca4f-6d86-4a65-8a2d-2660f0b808ae;arn:aws:braket:us-east-1:513796107385:quantum-task/7172dd0c-b4ac-47e8-a81f-36b54558457f;arn:aws:braket:us-east-1:513796107385:quantum-task/c9de1789-e580-4a85-bf22-3607edf875cf;arn:aws:braket:us-east-1:513796107385:quantum-task/f0a3635a-a821-433e-b8f8-e35ea6bdc7dd",
#     experiment_type="UQSD",
#     ibmq=False,
#     ionq=True,
# )
# """
# )
# exp0_plot(
#     num_points=20,
#     num_rounds=1,
#     shots=500,
#     n_qubit=2,
#     p1=0.5,
#     backend="ionq_aria_1",
#     job_id="arn:aws:braket:us-east-1:513796107385:quantum-task/b868ec82-67fb-4fac-8b39-5d95db04b591;arn:aws:braket:us-east-1:513796107385:quantum-task/dd3604b7-8886-4e26-a4e8-bc237ef5b5a4;arn:aws:braket:us-east-1:513796107385:quantum-task/a0f475cf-c9f1-4ce1-b100-64b5b40904a8;arn:aws:braket:us-east-1:513796107385:quantum-task/5f4e0f14-1c38-49ae-a7d7-075c5aa27c31;arn:aws:braket:us-east-1:513796107385:quantum-task/af42b0cd-eb43-433b-ba8c-1f29370bc53e;arn:aws:braket:us-east-1:513796107385:quantum-task/081f7168-57f7-470a-b712-d1a4b25abcaf;arn:aws:braket:us-east-1:513796107385:quantum-task/b263a9a8-1aba-4288-a844-e76f7e443bb4;arn:aws:braket:us-east-1:513796107385:quantum-task/7edd555d-92f9-4d78-b036-ceae3c6a5f81;arn:aws:braket:us-east-1:513796107385:quantum-task/66b59bd9-845e-4c7b-b9b4-2560bd85ce17;arn:aws:braket:us-east-1:513796107385:quantum-task/4379c941-c640-4b63-a2fa-0d2c6fb6faa3;arn:aws:braket:us-east-1:513796107385:quantum-task/1389260e-cdfa-4fa5-91da-beba92698453;arn:aws:braket:us-east-1:513796107385:quantum-task/bc6f4bce-6a1b-47ae-af9c-08ddf121bdc7;arn:aws:braket:us-east-1:513796107385:quantum-task/a5816c69-35f5-45fa-93d1-d295939c8bf9;arn:aws:braket:us-east-1:513796107385:quantum-task/f82cf425-085a-4b66-bcaf-73aafec04a15;arn:aws:braket:us-east-1:513796107385:quantum-task/338c9abd-1c65-42f5-9134-2c9d235862f6;arn:aws:braket:us-east-1:513796107385:quantum-task/7b677812-0993-4bee-a97a-486713aa8aa3;arn:aws:braket:us-east-1:513796107385:quantum-task/1d07f559-aa29-4636-9160-da3f877cfa06;arn:aws:braket:us-east-1:513796107385:quantum-task/319bdedf-ccf1-4430-9e3d-937084a6d39e;arn:aws:braket:us-east-1:513796107385:quantum-task/4e255cee-04e5-4b3d-bbad-20a213ef955e",
#     experiment_type="MED",
#     ibmq=False,
#     ionq=True,
# )
#
# exp0_plot(
#     num_points=20,
#     num_rounds=1,
#     shots=500,
#     n_qubit=2,
#     p1=0.5,
#     backend="ionq_aria_1",
#     job_id="arn:aws:braket:us-east-1:513796107385:quantum-task/8dfe136f-1941-41d3-abd9-8c8aeb958c9d;arn:aws:braket:us-east-1:513796107385:quantum-task/078e8670-7404-4dcf-a103-98047e884c9e;arn:aws:braket:us-east-1:513796107385:quantum-task/59f5988d-155d-4cfa-b642-44d292f8ee95;arn:aws:braket:us-east-1:513796107385:quantum-task/9de428b9-67fb-43d5-a809-5eb1aadd9809;arn:aws:braket:us-east-1:513796107385:quantum-task/a648baaa-2d87-4850-ba7c-28d605e21c7b;arn:aws:braket:us-east-1:513796107385:quantum-task/bf905598-fbe0-4c28-9efd-d454aa469803;arn:aws:braket:us-east-1:513796107385:quantum-task/ed408576-bc78-48ed-a3ad-118ae9419ab4;arn:aws:braket:us-east-1:513796107385:quantum-task/0030fae8-691b-4034-82e5-ac22de020622;arn:aws:braket:us-east-1:513796107385:quantum-task/c5f28701-9d22-4555-b13f-664ef8ab61bb;arn:aws:braket:us-east-1:513796107385:quantum-task/2102d1b1-c04c-4a19-9436-d73fe6617079;arn:aws:braket:us-east-1:513796107385:quantum-task/84d9ead5-0875-49c7-814f-fd59cd51c5ce;arn:aws:braket:us-east-1:513796107385:quantum-task/e57efaa3-15c9-468e-986e-28871736bc9a;arn:aws:braket:us-east-1:513796107385:quantum-task/01dce881-d94c-433b-947d-6197927d4b00;arn:aws:braket:us-east-1:513796107385:quantum-task/667185e7-2f3d-4241-87c1-3fa126ad5f3a;arn:aws:braket:us-east-1:513796107385:quantum-task/e52ee02f-c16e-4e0d-bedc-6503ef2ad519;arn:aws:braket:us-east-1:513796107385:quantum-task/4eef4ae1-944b-46b6-b642-e96dbb61955f;arn:aws:braket:us-east-1:513796107385:quantum-task/819614ee-c73a-4caf-871e-f8b92f05deba;arn:aws:braket:us-east-1:513796107385:quantum-task/fd544418-e638-42da-bbb5-33664ee7226c;arn:aws:braket:us-east-1:513796107385:quantum-task/1f9686b7-af81-4c08-afac-50c7a0902670",
#     experiment_type="UQSD",
#     ibmq=False,
#     ionq=True,
# )
# exp0_plot(
#     experiment_type="MED",
#     backend="ibmq_kolkata",
#     job_id="cr0r7nedvs8g008j94ng",
# )
