# nanoGPT with Azure

Originally forked from https://github.com/karpathy/nanoGPT and extended to be deployable on Azure ML

## Instructions

- Install the azure cli locally
- Login to the azure shell using `az login`
- If Azure workspace not yet created, create one by running `bash azure_deployment/create_workspace.sh`


## Local Development Setup Windows

Install conda environment

```
conda env create -f environment.yml
```

Or run 

`source start.sh`

Remove the old environment if something went wrong:

```
conda deactivate
conda env remove --name nano_gpt --all
```
