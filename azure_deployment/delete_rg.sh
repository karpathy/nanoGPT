#!/bin/bash

source ./azure_deployment/config.conf

echo "Resource group name: " $RESOURCE_GROUP

# Ask for confirmation before deleting the resource group
read -p "Are you sure you want to delete the resource group $RESOURCE_GROUP ? (y/n) " -n 1 -r

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "\n Deleting the resource group " $RESOURCE_GROUP
    az group delete --name $RESOURCE_GROUP --yes
fi
