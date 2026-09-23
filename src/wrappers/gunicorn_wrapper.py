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
#
# Gunicorn Python wrapper

import os
import sys

from gunicorn.app.base import BaseApplication


def get_application(file, function):
    """
    Dynamically import the ASGI application from the given file and function name.
    """
    module_path = file.replace("/", ".").rstrip(".py")
    module = __import__(module_path, fromlist=[function])
    return getattr(module, function)


class GunicornUvicornApplication(BaseApplication):
    """
    Custom Gunicorn application class that loads the ASGI application and Gunicorn configuration.
    """

    def __init__(self, application, options=None):
        self.application = application
        self.options = options or {}
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def run_visual_search():
    """Entry point for visual search service (called from pyproject.toml scripts)"""
    # Set default arguments for visual search service
    sys.argv = [
        sys.argv[0],
        "src/visual_search/main.py", 
        "app"
    ]
    main()


def main():
    """Main entry point for gunicorn wrapper"""
    print("Wrapper received arguments:", sys.argv)

    port = os.getenv("GUNICORN_PORT", "8000")
    workers = os.getenv("GUNICORN_WORKERS", "1")
    timeout = os.getenv("GUNICORN_TIMEOUT", "30")

    # Extract the file and function from the arguments
    file = sys.argv.pop(1)
    function = sys.argv.pop(1)

    # Validate the file extension
    stem, extension = file.split(".")
    assert extension == "py", f"File should be a Python (.py) file! Got {extension}."

    # Import the application
    app = get_application(file, function)

    # Define Gunicorn options
    options = {
        "bind": f"0.0.0.0:{port}",
        "workers": workers,
        "timeout": timeout,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "accesslog": "-",  # Log to stdout
        "errorlog": "-",  # Log to stderr
    }

    # Starting the Gunicorn server with Uvicorn workers
    GunicornUvicornApplication(app, options=options).run()


if __name__ == "__main__":
    main() 