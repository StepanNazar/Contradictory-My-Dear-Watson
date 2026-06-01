# Deployment Guide: Azure Container Apps

This guide describes how to deploy the Contradictory My Dear Watson application to Azure.

## Prerequisites
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed and configured (`az login`).
- [uv](https://github.com/astral-sh/uv) installed locally.

## Step 1: Local Model Preparation
Before building the Docker image, you must generate the model files locally.

```bash
uv run python scripts/train_stacked.py
```
This will create `models/current/` with 5 `.joblib` files.

## Step 2: Resource Configuration
Define your resource names:
```bash
RESOURCE_GROUP="WatsonGroup"
LOCATION="eastus"
ACR_NAME="watsonregistry$(date +%s)"
APP_NAME="watson-nli-app"
```

## Step 3: Create Azure Resources & Push Image
Azure CLI can handle ACR creation and image pushing in one go (or separately).

1. **Create Resource Group**:
   ```bash
   az group create --name $RESOURCE_GROUP --location $LOCATION
   ```

2. **Create Azure Container Registry**:
   ```bash
   az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true
   ```

3. **Build and Push to ACR**:
   ```bash
   az acr build --registry $ACR_NAME --image watson-nli:latest .
   ```

## Step 4: Deploy to Azure Container Apps
Deploy the image from ACR to a Container App. 

*Note: For 4GiB memory, Azure requires a specific CPU ratio (2.0 cores).*

1. **Create Container Apps Environment**:
   ```bash
   az containerapp env create \
     --name watson-env \
     --resource-group $RESOURCE_GROUP \
     --location $LOCATION
   ```

2. **Create the Container App**:
   ```bash
   az containerapp create \
     --name $APP_NAME \
     --resource-group $RESOURCE_GROUP \
     --environment watson-env \
     --image $ACR_NAME.azurecr.io/watson-nli:latest \
     --target-port 8000 \
     --ingress external \
     --registry-server $ACR_NAME.azurecr.io \
     --cpu 2.0 --memory 4.0Gi
   ```

## Step 5: Verify Deployment
Once the command completes, it will provide an FQDN (URL).
- Open the URL in your browser to see the UI.
- Use `/health` for status checks.
- Use `/docs` for Swagger UI.
