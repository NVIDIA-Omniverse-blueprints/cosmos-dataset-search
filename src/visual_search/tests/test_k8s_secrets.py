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

import unittest
from unittest.mock import patch, MagicMock
from kubernetes.client.rest import ApiException
from src.visual_search.v1.apis.k8s_secrets import get_k8s_secret, SecretsNotFoundError
import base64

class TestK8sSecrets(unittest.TestCase):

    @patch('src.visual_search.v1.apis.k8s_secrets.client.CoreV1Api')
    @patch('src.visual_search.v1.apis.k8s_secrets.config.load_incluster_config')
    def test_get_k8s_secret_success(self, mock_load_incluster_config, mock_core_v1_api):
        # Mock the Kubernetes API client
        mock_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_api_instance

        # Mock the secret data
        mock_secret = MagicMock()
        mock_secret.data = {
            "aws_access_key_id": base64.b64encode(b"test_access_key").decode('utf-8'),
            "aws_secret_access_key": base64.b64encode(b"test_secret_key").decode('utf-8'),
            "aws_region": base64.b64encode(b"test_region").decode('utf-8'),
            "endpoint_url": base64.b64encode(b"http://test.url").decode('utf-8'),
        }
        mock_api_instance.read_namespaced_secret.return_value = mock_secret

        # Call the function
        secret_data = get_k8s_secret("test-secret")

        # Assert the secret data is correct
        self.assertEqual(secret_data["aws_access_key_id"], "test_access_key")
        self.assertEqual(secret_data["aws_secret_access_key"], "test_secret_key")
        self.assertEqual(secret_data["aws_region"], "test_region")
        self.assertEqual(secret_data["endpoint_url"], "http://test.url")

    @patch('src.visual_search.v1.apis.k8s_secrets.client.CoreV1Api')
    @patch('src.visual_search.v1.apis.k8s_secrets.config.load_incluster_config')
    def test_get_k8s_secret_not_found(self, mock_load_incluster_config, mock_core_v1_api):
        # Mock the Kubernetes API client
        mock_api_instance = MagicMock()
        mock_core_v1_api.return_value = mock_api_instance

        # Simulate a 404 error
        mock_api_instance.read_namespaced_secret.side_effect = ApiException(status=404)

        # Call the function and assert it raises SecretsNotFoundError
        with self.assertRaises(SecretsNotFoundError):
            get_k8s_secret("non-existent-secret")

if __name__ == '__main__':
    unittest.main() 