import os
import platform
import sys

from packaging import version
from pathlib import Path
from typing import Tuple, Optional

import launch
from launch import is_installed, run, run_pip
import importlib.metadata

try:
    skip_install = getattr(launch.args, "skip_install")
except Exception:
    skip_install = getattr(launch, "skip_install", False)

python = sys.executable


def comparable_version(version: str) -> Tuple:
    return tuple(version.split("."))


def get_installed_version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def install():
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements-sdwebui.txt")
    if os.path.exists(req_file):
        mainpackage = 'sd-scripts'
        with open(req_file) as file:
            for package in file:
                try:
                    package = package.strip()
                    if package[0] in ["#", "-"]:
                        continue

                    if '==' in package:
                        package_name, package_version = package.split('==')
                        installed_version = get_installed_version(package_name)
                        if installed_version != package_version:
                            run_pip(f"install -U {package}", f"{mainpackage} requirement: changing {package_name} version from {installed_version} to {package_version}")
                    elif '>=' in package:
                        package_name, package_version = package.split('>=')
                        installed_version = get_installed_version(package_name)
                        if not installed_version or comparable_version(installed_version) < comparable_version(package_version):
                            run_pip(f"install -U {package}", f"{mainpackage} requirement: changing {package_name} version from {installed_version} to {package_version}")
                    elif not is_installed(package):
                        run_pip(f"install {package}", f"{mainpackage} requirement: {package}")
                except Exception as e:
                    print(f"Error: {e}")
                    print(f'Warning: Failed to install {package}, some preprocessors may not work.')


if not skip_install:
    install()
