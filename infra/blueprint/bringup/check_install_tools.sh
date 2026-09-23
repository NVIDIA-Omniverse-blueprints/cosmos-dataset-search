#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Check and install missing tools for CVDS deployment

# Create local bin directory if it doesn't exist
mkdir -p $HOME/bin
export PATH=$HOME/bin:$PATH

# Function to check if a command exists
check_command() {
    command -v "$1" &> /dev/null
}

# Function to add PATH to shell config
add_to_path() {
    if ! grep -q 'export PATH=$HOME/bin:$PATH' ~/.bashrc; then
        echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
    fi
}

echo "Checking required tools..."

# kubectl
if ! check_command kubectl; then
    echo "Installing kubectl..."
    KUBECTL_VERSION=$(curl -L -s https://dl.k8s.io/release/stable.txt)
    echo "  Downloading kubectl ${KUBECTL_VERSION}..."
    
    # Download kubectl binary
    curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
    
    # Download and verify checksum
    curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl.sha256"
    
    echo "  Verifying checksum..."
    if echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check --status; then
        echo "  ✓ Checksum verified"
        chmod +x kubectl
        mv kubectl $HOME/bin/
        rm kubectl.sha256
        add_to_path
    else
        echo "  ✗ ERROR: Checksum verification failed!"
        echo "  Removing potentially corrupted file..."
        rm -f kubectl kubectl.sha256
        exit 1
    fi
else
    echo "✓ kubectl is installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client -o yaml | grep gitVersion | head -1)"
fi

# helm
if ! check_command helm; then
    echo "Installing helm..."
    # Note: The official Helm installation script (get-helm-3) includes its own GPG signature verification
    # when downloading the Helm binary, so we don't need additional checksum verification here
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
    chmod 700 get_helm.sh
    HELM_INSTALL_DIR=$HOME/bin ./get_helm.sh --no-sudo
    rm get_helm.sh
    add_to_path
else
    echo "✓ helm is installed: $(helm version --short)"
fi

# eksctl
if ! check_command eksctl; then
    echo "Installing eksctl..."
    PLATFORM="$(uname -s)_amd64"
    
    # Save current directory
    SAVED_DIR=$(pwd)
    
    # Change to /tmp for download
    if ! cd /tmp; then
        echo "ERROR: Failed to change to /tmp directory"
        exit 1
    fi
    
    # Download eksctl and its checksum
    echo "  Downloading eksctl..."
    curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_${PLATFORM}.tar.gz"
    
    # Download checksums file
    curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_checksums.txt"
    
    # Verify checksum
    echo "  Verifying checksum..."
    if grep "eksctl_${PLATFORM}.tar.gz" eksctl_checksums.txt | sha256sum --check --status; then
        echo "  ✓ Checksum verified"
        tar xzf "eksctl_${PLATFORM}.tar.gz"
        mv eksctl $HOME/bin/
        rm -f "eksctl_${PLATFORM}.tar.gz" eksctl_checksums.txt
        add_to_path
    else
        echo "  ✗ ERROR: Checksum verification failed!"
        echo "  Removing potentially corrupted file..."
        rm -f "eksctl_${PLATFORM}.tar.gz" eksctl_checksums.txt eksctl
        cd "$SAVED_DIR"
        exit 1
    fi
    
    # Return to saved directory
    if ! cd "$SAVED_DIR"; then
        echo "WARNING: Failed to return to directory $SAVED_DIR"
    fi
else
    echo "✓ eksctl is installed: $(eksctl version)"
fi

# AWS CLI v2
if ! check_command aws || [[ $(aws --version 2>&1) != *"aws-cli/2"* ]]; then
    echo "Installing AWS CLI v2..."
    # Save current directory
    SAVED_DIR=$(pwd)
    
    # Download AWS CLI
    echo "  Downloading AWS CLI v2..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    
    # Note: Full GPG verification requires gpg to be installed and AWS public key imported
    # For now, we'll verify the download completed successfully
    
    # Change to /tmp with error handling
    if ! cd /tmp; then
        echo "ERROR: Failed to change to /tmp directory"
        exit 1
    fi
    
    # Verify the zip file is valid
    if unzip -t awscliv2.zip > /dev/null 2>&1; then
        echo "  ✓ ZIP file integrity verified"
        unzip -q awscliv2.zip
        ./aws/install -i $HOME/aws-cli -b $HOME/bin
    else
        echo "  ✗ ERROR: AWS CLI ZIP file is corrupted!"
        rm -f awscliv2.zip
        cd "$SAVED_DIR"
        exit 1
    fi
    
    # Return to saved directory with error handling
    if ! cd "$SAVED_DIR"; then
        echo "WARNING: Failed to return to directory $SAVED_DIR"
        # Continue but warn user
    fi
    
    rm -rf /tmp/awscliv2.zip /tmp/aws
    add_to_path
else
    echo "✓ aws is installed: $(aws --version)"
fi

# python3
if ! check_command python3; then
    echo "❌ python3 is not installed. Please install it using your system package manager:"
    echo "   Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "   RHEL/CentOS: sudo yum install python3 python3-pip"
else
    echo "✓ python3 is installed: $(python3 --version)"
fi

# git
if ! check_command git; then
    echo "❌ git is not installed. Please install it using your system package manager:"
    echo "   Ubuntu/Debian: sudo apt-get install git"
    echo "   RHEL/CentOS: sudo yum install git"
else
    echo "✓ git is installed: $(git --version)"
fi

# docker
if ! check_command docker; then
    echo "❌ docker is not installed. Please install Docker Desktop or Docker Engine from https://docs.docker.com/get-docker/"
else
    echo "✓ docker is installed: $(docker --version)"
fi

# jq
if ! check_command jq; then
    echo "Installing jq..."
    curl -L https://github.com/stedolan/jq/releases/download/jq-1.7/jq-linux64 -o $HOME/bin/jq
    chmod +x $HOME/bin/jq
    add_to_path
else
    echo "✓ jq is installed: $(jq --version)"
fi

# curl
if ! check_command curl; then
    echo "❌ curl is not installed. Please install it using your system package manager:"
    echo "   Ubuntu/Debian: sudo apt-get install curl"
    echo "   RHEL/CentOS: sudo yum install curl"
else
    echo "✓ curl is installed: $(curl --version | head -1)"
fi

# Update Helm repositories
if check_command helm; then
    echo ""
    echo "Updating Helm repositories..."
    helm repo add milvus https://zilliztech.github.io/milvus-helm/ || true
    helm repo add bitnami https://charts.bitnami.com/bitnami || true
    helm repo update
fi

echo ""
echo "Tool check complete!"
echo "If any tools show ❌, please install them manually using the provided commands."
echo ""
echo "After installing new tools, reload your shell configuration:"
echo "  source ~/.bashrc"
