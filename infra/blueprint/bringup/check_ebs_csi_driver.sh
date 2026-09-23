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

# Function to check addon status
check_addon_status() {
    echo "Checking the status of the aws-ebs-csi-driver addon..."
    addon_info=$(eksctl get addon --name aws-ebs-csi-driver --cluster $CLUSTER_NAME --region $AWS_REGION --output json)
    status=$(echo "$addon_info" | awk -F'"' '/Status/{print $4}')
    echo "Current status: $status"
}

# Main loop to wait for addon to become ACTIVE
while true; do
    check_addon_status

    if [[ "$status" == "ACTIVE" ]]; then
        echo "The aws-ebs-csi-driver addon is now ACTIVE."
        break
    elif [[ "$status" == "CREATING" ]]; then
        echo "The addon is still being created. Waiting for 30 seconds before checking again..."
        sleep 30
    else
        echo "There was an error or the addon is in an unexpected state: $status"
        exit 1
    fi
done
