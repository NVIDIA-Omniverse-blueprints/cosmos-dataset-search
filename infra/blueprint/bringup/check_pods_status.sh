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

# Namespace to check pods in
namespace="default"

# Function to check pod statuses
check_pods() {
    all_pods_running=0  # Assume all pods are running initially

    # Get all pods in the specified namespace
    pods=$(kubectl get pods -n "$namespace" -o jsonpath='{.items[*].metadata.name}')

    for pod in $pods; do
        # Get the status of the pod
        status=$(kubectl get pod "$pod" -n "$namespace" -o jsonpath='{.status.phase}')

        # Check if the pod is not in Running state
        if [ "$status" != "Running" ]; then
            all_pods_running=1  # Set to 1 if any pod is not running
            echo "Pod $pod is in $status state.\n"
        else
            echo -n "."
        fi
    done

        echo -e "\n"
    return $all_pods_running  # Return the status
}

# Main loop to wait for all pods to be in the Running state
while true; do
    check_pods
    result=$?  # Capture the return value of check_pods

    if [ $result -eq 0 ]; then
        echo "All pods are in the Running state. Process complete."
        break
    else
        echo "Waiting for all pods to be in the Running state..."
        sleep 10 # Wait for 10 seconds before checking again
    fi
done
