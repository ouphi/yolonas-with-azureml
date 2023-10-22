# Train YOLO-NAS with AzureML

## Example of training using Knowledge distillation

The student is a small model resnet18 with 11M parameters pre-trained on cifar10 
and the teacher is a [beit_base_patch16_224](https://github.com/microsoft/unilm/tree/master/beit) model with 87M parameters pre-trained on ImageNet-22k. 

## Prerequisites

- az cli and [az ml extension installed](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?view=azureml-api-2&tabs=public)

## Set up workspace config

```bash
az configure --defaults workspace=<your-workspace>
az configure --defaults group=<your-resource-group>
```

## Create a compute cluster

[Create an AzureML compute cluster](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?view=azureml-api-2&tabs=azure-cli#create) with enough resource to run your training. 

## Create an AzureML environment

```bash
az ml environment create -f azureml/environment.yaml
```

## Run the training

```bash
az ml job create -f azureml/job.yaml 
```

