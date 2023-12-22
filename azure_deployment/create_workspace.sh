# Run this script in the Azure cloud shell to create a workspace and compute resources

#! /usr/bin/sh

# Make sure the Azure ml CLI is installed
az extension add -n ml

echo "Verify version of az ml CLI, it should be >=2.22.0"
az extension show --name ml --output table

# Set the necessary variables
# Read Resource Group name from the config file
source ./azure_deployment/config.conf
echo "Resource group name: " $RESOURCE_GROUP

RESOURCE_PROVIDER="Microsoft.MachineLearning"
WORKSPACE_NAME="ml-nano-gpt"
COMPUTE_INSTANCE="ci-nano-gpt"
COMPUTE_CLUSTER="aml-cluster-nano-gpt"

# Register the Azure Machine Learning resource provider in the subscription
echo "Register the Machine Learning resource provider:"
az provider register --namespace $RESOURCE_PROVIDER

# Create the resource group and workspace and set to default
echo "Create a resource group and set as default:"
az group create --name $RESOURCE_GROUP --location $REGION
az configure --defaults group=$RESOURCE_GROUP

echo "Sleeping for 30 seconds to allow the resource group to be created..."
sleep 30

echo "Create an Azure Machine Learning workspace:"
az ml workspace create --name $WORKSPACE_NAME 
az configure --defaults workspace=$WORKSPACE_NAME 

echo "Sleeping for 30 seconds to allow the ml workspace to be created..."
sleep 30

# Create compute instance
echo "Creating a compute instance with name: " $COMPUTE_INSTANCE
az ml compute create --name ${COMPUTE_INSTANCE} --size ${COMPUTE_VM} --type ComputeInstance 

# Create compute cluster
echo "Creating a compute cluster with name: " $COMPUTE_CLUSTER
az ml compute create --name ${COMPUTE_CLUSTER} --size ${COMPUTE_VM} --max-instances 2 --type AmlCompute 

# Create data assets for NLP next token prediction datasets (pretraining)
echo "Create training data asset:"
az ml data create --type uri_file --name "shakespeare-corpus" --path ./azure_deployment/data/shakespeare.txt

# Set idle shutdown timer to 15 minutes
echo "Setting idle shutdown timer to 15 minutes for compute instance: " $COMPUTE_INSTANCE
az ml compute update --name ${COMPUTE_INSTANCE} --idle-shutdown-timeout 900