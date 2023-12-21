# Run this script in the Azure cloud shell to create a workspace and compute resources

#! /usr/bin/sh

# Set the necessary variables
RESOURCE_GROUP="rg-nano-gpt"
RESOURCE_PROVIDER="Microsoft.MachineLearning"
REGIONS=("eastus" "westus" "centralus" "northeurope" "westeurope")
RANDOM_REGION=${REGIONS[$RANDOM % ${#REGIONS[@]}]}
WORKSPACE_NAME="mlw-nano-gpt"
COMPUTE_INSTANCE="ci-nano-gpt"
COMPUTE_CLUSTER="aml-cluster-nano-gpt"

# Register the Azure Machine Learning resource provider in the subscription
echo "Register the Machine Learning resource provider:"
az provider register --namespace $RESOURCE_PROVIDER

# Create the resource group and workspace and set to default
echo "Create a resource group and set as default:"
az group create --name $RESOURCE_GROUP --location $RANDOM_REGION
az configure --defaults group=$RESOURCE_GROUP

echo "Create an Azure Machine Learning workspace:"
az ml workspace create --name $WORKSPACE_NAME 
az configure --defaults workspace=$WORKSPACE_NAME 

# Create compute instance
echo "Creating a compute instance with name: " $COMPUTE_INSTANCE
az ml compute create --name ${COMPUTE_INSTANCE} --size STANDARD_DS11_V2 --type ComputeInstance 

# Create compute cluster
echo "Creating a compute cluster with name: " $COMPUTE_CLUSTER
az ml compute create --name ${COMPUTE_CLUSTER} --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute 

# Create data assets
echo "Create training data asset:"
az ml data create --type uri_file --name "diabetes-data" --path ./data/diabetes.csv 