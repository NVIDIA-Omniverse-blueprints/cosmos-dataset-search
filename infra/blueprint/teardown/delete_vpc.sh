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

function delete_vpc() {
  : "${CLUSTER_NAME:?CLUSTER_NAME must identify the cluster being removed}"
  : "${AWS_REGION:?AWS_REGION must identify the cluster region}"
  ###################### Delete VPC ######################
  VPC_NAME="eksctl-${CLUSTER_NAME}-cluster/VPC"

  echo "Looking up VPC ID for Name=$VPC_NAME ..."
  VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=tag:Name,Values=${VPC_NAME}" \
    --query "Vpcs[0].VpcId" \
    --output text)

  if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
    echo "No VPC found with Name=$VPC_NAME. Exiting."
    return 0
  fi

  echo "Found VPC ID: $VPC_ID"

  # 1) Delete NAT Gateways, then wait for them to vanish
  echo "Deleting NAT Gateways..."
  NAT_GWS=$(aws ec2 describe-nat-gateways \
    --filter "Name=vpc-id,Values=$VPC_ID" \
    --query "NatGateways[].NatGatewayId" \
    --output text | tr '\t' '\n')
  if [ -n "$NAT_GWS" ]; then
    for NGW in $NAT_GWS; do
      echo " - Deleting NAT Gateway $NGW"
      aws ec2 delete-nat-gateway --nat-gateway-id "$NGW"
    done

    # NAT gateway deletion isn’t instantaneous; we need to poll
    echo "Waiting for NAT Gateways to reach State=deleted..."
    while true; do
      # Grab any NAT gateways that are still not in "deleted" state
      STILL_DELETING=$(aws ec2 describe-nat-gateways \
        --filter "Name=vpc-id,Values=$VPC_ID" \
        --query "NatGateways[?State!='deleted'].NatGatewayId" \
        --output text | tr '\t' '\n')

      # If the result is empty or "None", they’ve all been fully deleted
      if [ -z "$STILL_DELETING" ] || [ "$STILL_DELETING" = "None" ]; then
        break
      fi

      echo "   Still deleting: $STILL_DELETING ... sleeping 15s"
      sleep 15
    done
  fi

  echo "Releasing unattached EIPs owned by cluster $CLUSTER_NAME..."
  EIP_IDS=$(aws ec2 describe-addresses \
    --region "$AWS_REGION" \
    --filters "Name=tag:aws:cloudformation:stack-name,Values=eksctl-${CLUSTER_NAME}-cluster" \
    --query "Addresses[?AssociationId==null].AllocationId" \
    --output text || true)

  if [ -n "$EIP_IDS" ] && [ "$EIP_IDS" != "None" ]; then
    for eip in $EIP_IDS; do
      echo "Releasing unattached EIP: $eip"
      aws ec2 release-address --region "$AWS_REGION" --allocation-id "$eip" || true
    done
  else
    echo "No unattached EIPs to release."
  fi

  echo "Cleaning up subnets and their route tables..."
  SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "Subnets[].SubnetId" \
    --output text | tr '\t' '\n')

  if [ -n "$SUBNETS" ] && [ "$SUBNETS" != "None" ]; then
    for SUBNET in $SUBNETS; do
      echo "Subnet: $SUBNET"

      # Find route tables associated with this subnet
      # We'll parse out AssocId, Main, and the RTB ID for each association
      ASSOCS=$(aws ec2 describe-route-tables \
        --filters "Name=association.subnet-id,Values=$SUBNET" \
        --query "RouteTables[].Associations[].{AssocId:RouteTableAssociationId, Main:Main, RtbId:RouteTableId}" \
        --output text)

      # Disassociate & delete non-main route tables
      while read -r ASSOC_ID MAIN RTB_ID; do
        # Skip any blank lines
        [ -z "$MAIN" ] && continue

        echo "Found association: AssocId=$ASSOC_ID, Main=$MAIN, RtbId=$RTB_ID"
        if [ "$MAIN" = "False" ]; then
          echo "  - Disassociating from route table $RTB_ID (Assoc ID: $ASSOC_ID)"
          aws ec2 disassociate-route-table --association-id "$ASSOC_ID"

          echo "  - Deleting route table $RTB_ID"
          aws ec2 delete-route-table --route-table-id "$RTB_ID" || true
        fi
      done <<< "$ASSOCS"

      # Find and delete any ENIs in this subnet
      ENIS=$(aws ec2 describe-network-interfaces \
        --filters "Name=subnet-id,Values=$SUBNET" \
        --query "NetworkInterfaces[].NetworkInterfaceId" \
        --output text | tr '\t' '\n')

      if [ -n "$ENIS" ] && [ "$ENIS" != "None" ]; then
        for ENI in $ENIS; do
          echo "  - Found ENI $ENI in subnet $SUBNET"
          # If the ENI is attached, detach it
          ATTACH_ID=$(aws ec2 describe-network-interfaces \
            --network-interface-ids "$ENI" \
            --query "NetworkInterfaces[0].Attachment.AttachmentId" \
            --output text)

          if [ "$ATTACH_ID" != "None" ]; then
            echo "    * Detaching ENI $ENI (Attachment: $ATTACH_ID)"
            aws ec2 detach-network-interface --attachment-id "$ATTACH_ID" --force || true
            # Optionally wait for the interface to become detached
          fi

          echo "    * Deleting ENI $ENI"
          aws ec2 delete-network-interface --network-interface-id "$ENI" || true
        done
      fi

      # Finally, remove the subnet itself
      echo "  - Deleting subnet $SUBNET"
      aws ec2 delete-subnet --subnet-id "$SUBNET" || true
    done
  fi

  # 2) Delete any VPC Endpoints
  echo "Deleting VPC endpoints..."
  ENDPOINTS=$(aws ec2 describe-vpc-endpoints \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "VpcEndpoints[].VpcEndpointId" \
    --output text | tr '\t' '\n')
  if [ -n "$ENDPOINTS" ] && [ "$ENDPOINTS" != "None" ]; then
    echo " - Deleting endpoints: $ENDPOINTS"
    aws ec2 delete-vpc-endpoints --vpc-endpoint-ids $ENDPOINTS
  fi

  # 4) Detach and delete any Internet Gateways
  echo "Removing Internet Gateways..."
  IGWS=$(aws ec2 describe-internet-gateways \
    --filters "Name=attachment.vpc-id,Values=$VPC_ID" \
    --query "InternetGateways[].InternetGatewayId" \
    --output text | tr '\t' '\n')
  if [ -n "$IGWS" ] && [ "$IGWS" != "None" ]; then
    for IGW in $IGWS; do
      echo " - Detaching IGW $IGW from $VPC_ID"
      aws ec2 detach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC_ID" || true
      echo " - Deleting IGW $IGW"
      aws ec2 delete-internet-gateway --internet-gateway-id "$IGW" || true
    done
  fi

  # 5) Delete custom Network ACLs (the default ACL can't be deleted)
  echo "Deleting custom Network ACLs..."
  ACLS=$(aws ec2 describe-network-acls \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "NetworkAcls[].NetworkAclId" \
    --output text | tr '\t' '\n')
  if [ -n "$ACLS" ] && [ "$ACLS" != "None" ]; then
    for ACL in $ACLS; do
      IS_DEFAULT=$(aws ec2 describe-network-acls \
        --network-acl-ids "$ACL" \
        --query "NetworkAcls[0].IsDefault" \
        --output text)
      if [ "$IS_DEFAULT" != "True" ]; then
        echo " - Deleting Network ACL $ACL"
        aws ec2 delete-network-acl --network-acl-id "$ACL" || true
      fi
    done
  fi
  # 7) Delete non-default Security Groups
  echo "Deleting Security Groups (except default)..."
  SGS=$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[].GroupId" \
    --output text | tr '\t' '\n')
  if [ -n "$SGS" ] && [ "$SGS" != "None" ]; then
    for SG in $SGS; do
      SG_NAME=$(aws ec2 describe-security-groups \
        --group-ids "$SG" \
        --query "SecurityGroups[0].GroupName" \
        --output text)
      if [ "$SG_NAME" != "default" ]; then
        echo " - Deleting security group $SG ($SG_NAME)"
        aws ec2 delete-security-group --group-id "$SG" || true
      fi
    done
  fi

  # 8) Finally, delete the VPC
  echo "Deleting VPC $VPC_ID..."
  aws ec2 delete-vpc --vpc-id "$VPC_ID" || true

  echo "Done. VPC $VPC_ID (Name=$VPC_NAME) has been cleaned up."
  return 0
}
