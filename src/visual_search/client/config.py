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

"""Utilities to manage CDS profiles."""

import configparser
import os
import shutil
from typing import Dict, Final, Optional

from pydantic import BaseModel

CONFIG_FILE: Final = os.path.expanduser("~/.config/cds/config")
OLD_CONFIG_FILE: Final = os.path.expanduser("~/.config/vius/config")
CDS_API_ENDPOINT: Final = "api_endpoint"
CDS_AUTH_KEY: Final = "auth_key"
CDS_AUTH_ENDPOINT: Final = "auth_endpoint"
DEFAULT: Final = "default"


class Profile(BaseModel):
    api_endpoint: str
    auth_key: Optional[str] = None
    auth_endpoint: Optional[str] = None

    class Config:
        frozen = True


class ProfilesConfig(BaseModel):
    profiles: Dict[str, Profile]

    class Config:
        frozen = True

    def get_profile(self, name: str) -> Profile:
        if name not in self.profiles:
            raise KeyError(f"Profile {name} is not available in `~/.config/cds/config`")
        return self.profiles[name]


def configure_credentials(profile: str = DEFAULT) -> configparser.ConfigParser:
    """Configure credentials for CDS client.
    
    :param profile: Profile name to configure (default: 'default')
    """

    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    config = configparser.ConfigParser()

    if os.path.exists(CONFIG_FILE):
        print(f"Config file exists. Will update/add profile '{profile}'.")
        config.read(CONFIG_FILE)
    
    # Create section if it doesn't exist
    if profile not in config:
        config[profile] = {}

    api_endpoint = input(
        f"Enter {CDS_API_ENDPOINT} for profile '{profile}' (base URL of the CDS API): ",
    )
    if not api_endpoint:
        raise ValueError(f"{CDS_API_ENDPOINT} cannot be None!")
    auth_endpoint = None
    auth_key = None

    config[profile][CDS_API_ENDPOINT] = api_endpoint
    if auth_endpoint and auth_key:
        config[profile][CDS_AUTH_KEY] = auth_key
        config[profile][CDS_AUTH_ENDPOINT] = auth_endpoint
    else:
        config[profile].pop(CDS_AUTH_KEY, None)
        config[profile].pop(CDS_AUTH_ENDPOINT, None)

    with open(CONFIG_FILE, "w") as configfile:
        config.write(configfile)

    print(f"Config for profile '{profile}' has been saved to {CONFIG_FILE}")
    return config


def load_config() -> ProfilesConfig:
    """Loads profiles. Supports migration from old vius config."""

    # Migrate old vius config if it exists
    if not os.path.exists(CONFIG_FILE) and os.path.exists(OLD_CONFIG_FILE):
        print(f"Migrating config from {OLD_CONFIG_FILE} to {CONFIG_FILE}")
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        shutil.copy2(OLD_CONFIG_FILE, CONFIG_FILE)
        print(f"Migration complete. Old config preserved at {OLD_CONFIG_FILE}")

    if not os.path.exists(CONFIG_FILE):
        config = configure_credentials()
    else:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)

    return ProfilesConfig(
        **{
            "profiles": {
                section: dict(config.items(section)) for section in config.sections()
            }
        }
    )
