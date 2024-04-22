import os
import numpy as np
import matplotlib.pyplot as plt


def retrieve_job_from_braket(
    experiment_type="UQSD",
    num_points=20,
    num_rounds=1,
    shots=500,
    n_qubit=2,
    p1=0.5,
    ionq=False,
    backend="Aria 1",
    job_id="cqk9jybt65cg0087v79g",
):
    # TODO
    # To use the code outside of AWS,
    # 1. Setup ~/.aws/config and ~/.aws/credentials
    # 2. If no access keys, create access key with "aws iam create-access-key" in CloudShell

    from qiskit_braket_provider import BraketProvider
    from braket.jobs import save_job_result, load_job_result

    service = BraketProvider().get_backend(backend)
    job = service.retrieve_job(job_id)

    # Assume the job is retrieved
    print(job.status())
    result = job.result()
    os.environ["AMZN_BRAKET_JOB_RESULTS_DIR"] = f"./results_aria1_{experiment_type}"
    save_job_result(result.get_counts())
    return job


def load_aria1_results(experiment_type="UQSD"):
    from braket.jobs import load_job_result

    result_json = load_job_result(f"results_aria1_{experiment_type}/results.json")
    result = result_json["result"]
    return result


def retrieve_job_from_ibmq(
    experiment_type="UQSD",
    num_points=20,
    num_rounds=1,
    shots=500,
    n_qubit=2,
    p1=0.5,
    ionq=False,
    backend="ibm_osaka",
    job_id="cqk9jybt65cg0087v79g",
):
    # TODO

    from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Sampler

    service = QiskitRuntimeService(channel="ibm_quantum")
    job = service.job(job_id)

    # Assume the job is retrieved
    print(job.status())
    result = job.result()
    # import json
    # with open(f"{backend}_{experiment_type}.json", "w") as outfile:
    #     json.dump(result.results().to_dict(), outfile)
    # from_dict() to_dict() json.load()
    # counts_list = job.result().get_counts()
    return job


if __name__ == "__main__":
    job = retrieve_job_from_braket(
        experiment_type="MED",
        job_id="arn:aws:braket:us-east-1:513796107385:quantum-task/b868ec82-67fb-4fac-8b39-5d95db04b591;arn:aws:braket:us-east-1:513796107385:quantum-task/dd3604b7-8886-4e26-a4e8-bc237ef5b5a4;arn:aws:braket:us-east-1:513796107385:quantum-task/a0f475cf-c9f1-4ce1-b100-64b5b40904a8;arn:aws:braket:us-east-1:513796107385:quantum-task/5f4e0f14-1c38-49ae-a7d7-075c5aa27c31;arn:aws:braket:us-east-1:513796107385:quantum-task/af42b0cd-eb43-433b-ba8c-1f29370bc53e;arn:aws:braket:us-east-1:513796107385:quantum-task/081f7168-57f7-470a-b712-d1a4b25abcaf;arn:aws:braket:us-east-1:513796107385:quantum-task/b263a9a8-1aba-4288-a844-e76f7e443bb4;arn:aws:braket:us-east-1:513796107385:quantum-task/7edd555d-92f9-4d78-b036-ceae3c6a5f81;arn:aws:braket:us-east-1:513796107385:quantum-task/66b59bd9-845e-4c7b-b9b4-2560bd85ce17;arn:aws:braket:us-east-1:513796107385:quantum-task/4379c941-c640-4b63-a2fa-0d2c6fb6faa3;arn:aws:braket:us-east-1:513796107385:quantum-task/1389260e-cdfa-4fa5-91da-beba92698453;arn:aws:braket:us-east-1:513796107385:quantum-task/bc6f4bce-6a1b-47ae-af9c-08ddf121bdc7;arn:aws:braket:us-east-1:513796107385:quantum-task/a5816c69-35f5-45fa-93d1-d295939c8bf9;arn:aws:braket:us-east-1:513796107385:quantum-task/f82cf425-085a-4b66-bcaf-73aafec04a15;arn:aws:braket:us-east-1:513796107385:quantum-task/338c9abd-1c65-42f5-9134-2c9d235862f6;arn:aws:braket:us-east-1:513796107385:quantum-task/7b677812-0993-4bee-a97a-486713aa8aa3;arn:aws:braket:us-east-1:513796107385:quantum-task/1d07f559-aa29-4636-9160-da3f877cfa06;arn:aws:braket:us-east-1:513796107385:quantum-task/319bdedf-ccf1-4430-9e3d-937084a6d39e;arn:aws:braket:us-east-1:513796107385:quantum-task/4e255cee-04e5-4b3d-bbad-20a213ef955e",
    )
    job.result().get_counts()
    retrieve_job_from_braket(
        experiment_type="UQSD",
        job_id="arn:aws:braket:us-east-1:513796107385:quantum-task/8dfe136f-1941-41d3-abd9-8c8aeb958c9d;arn:aws:braket:us-east-1:513796107385:quantum-task/078e8670-7404-4dcf-a103-98047e884c9e;arn:aws:braket:us-east-1:513796107385:quantum-task/59f5988d-155d-4cfa-b642-44d292f8ee95;arn:aws:braket:us-east-1:513796107385:quantum-task/9de428b9-67fb-43d5-a809-5eb1aadd9809;arn:aws:braket:us-east-1:513796107385:quantum-task/a648baaa-2d87-4850-ba7c-28d605e21c7b;arn:aws:braket:us-east-1:513796107385:quantum-task/bf905598-fbe0-4c28-9efd-d454aa469803;arn:aws:braket:us-east-1:513796107385:quantum-task/ed408576-bc78-48ed-a3ad-118ae9419ab4;arn:aws:braket:us-east-1:513796107385:quantum-task/0030fae8-691b-4034-82e5-ac22de020622;arn:aws:braket:us-east-1:513796107385:quantum-task/c5f28701-9d22-4555-b13f-664ef8ab61bb;arn:aws:braket:us-east-1:513796107385:quantum-task/2102d1b1-c04c-4a19-9436-d73fe6617079;arn:aws:braket:us-east-1:513796107385:quantum-task/84d9ead5-0875-49c7-814f-fd59cd51c5ce;arn:aws:braket:us-east-1:513796107385:quantum-task/e57efaa3-15c9-468e-986e-28871736bc9a;arn:aws:braket:us-east-1:513796107385:quantum-task/01dce881-d94c-433b-947d-6197927d4b00;arn:aws:braket:us-east-1:513796107385:quantum-task/667185e7-2f3d-4241-87c1-3fa126ad5f3a;arn:aws:braket:us-east-1:513796107385:quantum-task/e52ee02f-c16e-4e0d-bedc-6503ef2ad519;arn:aws:braket:us-east-1:513796107385:quantum-task/4eef4ae1-944b-46b6-b642-e96dbb61955f;arn:aws:braket:us-east-1:513796107385:quantum-task/819614ee-c73a-4caf-871e-f8b92f05deba;arn:aws:braket:us-east-1:513796107385:quantum-task/fd544418-e638-42da-bbb5-33664ee7226c;arn:aws:braket:us-east-1:513796107385:quantum-task/1f9686b7-af81-4c08-afac-50c7a0902670",
    )
