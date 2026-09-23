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

import base64
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from src.visual_search.common.exceptions import SecretsNotFoundError


def get_k8s_secret(storage_secrets: str, namespace: str = "default") -> dict:
    """
    Retrieve a secret from Kubernetes and decode its data.

    :param storage_secrets: The name of the Kubernetes secret.
    :param namespace: The Kubernetes namespace where the secret is located.
    :return: A dictionary containing the decoded secret data.
    :raises SecretsNotFoundError: If the secret is not found.
    """
    # Load Kubernetes configuration
    config.load_incluster_config()

    # Create a Kubernetes API client
    v1 = client.CoreV1Api()

    try:
        # Retrieve the secret from Kubernetes
        secret = v1.read_namespaced_secret(name=storage_secrets, namespace=namespace)
    except ApiException as e:
        if e.status == 404:
            raise SecretsNotFoundError(f"Secret '{storage_secrets}' not found in Kubernetes.")
        else:
            raise

    # Decode the secret data
    aws_access_key_id = base64.b64decode(secret.data.get("aws_access_key_id", "")).decode('utf-8')
    aws_secret_access_key = base64.b64decode(secret.data.get("aws_secret_access_key", "")).decode('utf-8')
    aws_region = base64.b64decode(secret.data.get("aws_region", "")).decode('utf-8')
    endpoint_url = base64.b64decode(secret.data.get("endpoint_url", "")).decode('utf-8')
    
    # Decode session token if present (for temporary credentials)
    aws_session_token = None
    if secret.data.get("aws_session_token"):
        aws_session_token = base64.b64decode(secret.data.get("aws_session_token")).decode('utf-8')
        if not aws_session_token.strip():
            aws_session_token = None

    result = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_region": aws_region,
        "endpoint_url": endpoint_url,
    }

    # Only add session token if it exists and is non-empty
    # This supports both permanent credentials and temporary credentials
    if aws_session_token:
        result["aws_session_token"] = aws_session_token
    
    return result 