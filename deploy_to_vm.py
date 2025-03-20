import json
from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from azureml.core import Workspace, Environment, Model, InferenceConfig
from azureml.core.compute import ComputeTarget, RemoteCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.webservice import LocalWebservice

# ----- Configuration -----
SUBSCRIPTION_ID = "<your-subscription-id>"
RESOURCE_GROUP = "<your-resource-group>"
WORKSPACE_NAME = "<your-workspace-name>"

MODEL_NAME = "Llama-3.3-70B-Instruct"
MODEL_VERSION = 1

VM_NAME = "<your-vm-name>"
VM_RESOURCE_GROUP = "<your-vm-resource-group>"
VM_USERNAME = "<your-vm-admin-username>"
VM_PASSWORD = "<your-vm-admin-password>"  # or specify SSH key in attach_configuration

# ----- 1. Authentication -----
credential = InteractiveBrowserCredential()
ml_client = MLClient(credential, subscription_id=SUBSCRIPTION_ID,
                     resource_group_name=RESOURCE_GROUP, workspace_name=WORKSPACE_NAME)
ws = Workspace(subscription_id=SUBSCRIPTION_ID, resource_group=RESOURCE_GROUP, workspace_name=WORKSPACE_NAME)

# ----- 2. Retrieve Model from Model Catalog -----
model = ml_client.models.get(name=MODEL_NAME, version=MODEL_VERSION)
print(f"Retrieved model: {model.name} v{model.version}")
# Download model (optional)
download_path = "./model_download"
ml_client.models.download(name=MODEL_NAME, version=MODEL_VERSION, download_path=download_path)
print(f"Model downloaded to {download_path}")

# Register downloaded model to workspace (optional if model not already in workspace registry)
registered_model = Model.register(workspace=ws, model_path=download_path, model_name="Llama3.3-70B")
print(f"Model registered in workspace: {registered_model.name}, id: {registered_model.id}")

# ----- 3. Attach VM Compute Target -----
compute_name = "attached-vm-compute"
VM_RESOURCE_ID = (f"/subscriptions/{SUBSCRIPTION_ID}/resourceGroups/{VM_RESOURCE_GROUP}"
                  f"/providers/Microsoft.Compute/virtualMachines/{VM_NAME}")
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print(f"Compute target '{compute_name}' already exists.")
except ComputeTargetException:
    attach_config = RemoteCompute.attach_configuration(resource_id=VM_RESOURCE_ID,
                                                      ssh_port=22,
                                                      username=VM_USERNAME,
                                                      password=VM_PASSWORD)
    compute_target = ComputeTarget.attach(ws, compute_name, attach_config)
    compute_target.wait_for_completion(show_output=True)
    print(f"Attached compute target: {compute_name}")

# ----- 4. Setup Environment and Inference Config -----
env = Environment(name="llama-env")
env.docker.enabled = True
env.docker.base_image = "mcr.microsoft.com/azureml/curated/minimal-ubuntu:latest"
env.python.conda_dependencies.add_pip_package("transformers==4.33.2")
env.python.conda_dependencies.add_pip_package("torch>=2.0")
env.python.conda_dependencies.add_pip_package("accelerate")
env.register(workspace=ws)

inference_config = InferenceConfig(entry_script="score.py", environment=env)
deployment_config = LocalWebservice.deploy_configuration(port=8890)

# ----- 5. Deploy Model to VM -----
service_name = "llama-service"
service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[registered_model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       deployment_target=compute_target,
                       overwrite=True)
service.wait_for_deployment(show_output=True)
print(f"Service state: {service.state}")

# ----- 6. Test the Deployment -----
if service.state == "Healthy":
    test_input = {"prompt": "Hello, my name is LLaMA and"}
    result = service.run(input_data=json.dumps(test_input))
    print("Inference result:", result)
    # You can also test via HTTP if needed:
    # vm_ip = "<your-vm-ip>"
    # resp = requests.post(f"http://{vm_ip}:8890/score", json=test_input)
    # print(resp.status_code, resp.text)
else:
    print("Deployment failed or service is unhealthy. Check logs:")
    print(service.get_logs())
