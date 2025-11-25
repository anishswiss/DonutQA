from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment
from azure.identity import DefaultAzureCredential
from datetime import datetime

# Connect to AML workspace
ml_client = MLClient.from_config(DefaultAzureCredential())


# Define environment
donut_env = Environment(
    name="donut-lora-env",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7:latest",
    conda_file="environment.yaml"
)

# Register environment
ml_client.environments.create_or_update(donut_env)

# Create endpoint
endpoint_name = f"donutqa-endpoint-{datetime.now().strftime('%m%d%H%M')}"
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Deploy model
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model="donutQA:1",           # or use model object from ml_client.models.get(...)
    environment=donut_env,
    code_path="src",              # folder containing score.py
    entry_script="score.py",
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# Route traffic
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"Endpoint {endpoint_name} deployed successfully!")
