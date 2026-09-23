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

FROM nvcr.io/nvidia/ai-workbench/python-cuda122:1.0.3

RUN apt-get update && apt-get install -y unzip gettext less

WORKDIR /workspace
RUN mkdir packages
WORKDIR /workspace/packages

# aws-cli v2.17.13
RUN curl -LO "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-2.17.13.zip"
RUN unzip awscli-exe-linux-x86_64-2.17.13.zip
RUN ./aws/install

# eksctl latest (using recommended download method)
RUN curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz"
RUN tar -xzf eksctl_Linux_amd64.tar.gz
RUN mv eksctl /usr/local/bin

# kubectl v1.30.1
RUN curl -LO "https://dl.k8s.io/release/v1.30.1/bin/linux/amd64/kubectl"
RUN install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# helm v3.17.0
RUN curl -LO "https://get.helm.sh/helm-v3.17.0-linux-amd64.tar.gz"
RUN tar -xzf helm-v3.17.0-linux-amd64.tar.gz
RUN mv linux-amd64/helm /usr/local/bin/helm

# Python v3.10 and pip v22.02 is already installed

# s5cmd v2.3.0
RUN curl -LO "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_linux_amd64.deb"
RUN apt install ./s5cmd_2.3.0_linux_amd64.deb

# kaggle v1.6.17
RUN pip install kaggle==1.6.17

# python magic v0.4.27
RUN apt-get install -y libmagic-dev
RUN pip install python-magic==0.4.27

# Update/remove some packages to avoid security vulnerabilities
RUN pip uninstall -y jupyterlab jupyter_server jupyter-lsp
RUN pip install tornado==6.4.2

# Generate secret encryption keys with the reviewed application dependency.
RUN pip install cryptography==50.0.1
RUN python3 -c "import cryptography; from cryptography.fernet import Fernet; \
    assert cryptography.__version__ == '50.0.1'; \
    codec = Fernet(Fernet.generate_key()); \
    assert codec.decrypt(codec.encrypt(b'build-check')) == b'build-check'"

# Used by the pipeline model-seed job to fetch an immutable HF snapshot.
RUN pip install huggingface-hub==0.34.3

# Change directory and delete downloaded temporary files
WORKDIR /workspace
RUN rm -rf /workspace/packages
