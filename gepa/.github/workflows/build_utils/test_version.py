# Obtained from https://github.com/stanfordnlp/dspy/blob/main/.github/workflows/build_utils/test_version.py
import sys
from datetime import datetime

import requests
import semver
from packaging.version import Version as PyPIVersion


def get_latest_version(package_name, tag_version, pypi_url="https://pypi.org"):
    # Returns latest version, and T/F as to whether it needs to be incremented
    response = requests.get(f"{pypi_url}/pypi/{package_name}/json")
    if response.status_code == 200:
        data = response.json()
        # Flatten the list of files for all releases and get the latest upload
        all_uploads = [
            (release["upload_time"], release["filename"], version)
            for version, releases in data["releases"].items()
            for release in releases
        ]
        # If a release with tag_version does not exist, that is the latest version
        # Then increment is False, as no need to increment the version
        tag_release_exists = any(upload for upload in all_uploads if upload[2] == tag_version)
        if not (tag_release_exists):
            return tag_version, False
        # Else, get the latest release version, and set increment to True
        else:
            # Sort all uploads by upload time in descending order
            latest_upload = max(all_uploads, key=lambda x: datetime.fromisoformat(x[0].rstrip("Z")))
            return latest_upload[2], True

    elif response.status_code == 404:
        # If no existing releases can get a 404
        return tag_version, False
    return None, None


def increment_version(curr_version):
    pypi_v = PyPIVersion(curr_version)
    if pypi_v.pre:
        pre = "".join([str(i) for i in pypi_v.pre])
        parsed_v = semver.Version(*pypi_v.release, pre)
    else:
        parsed_v = semver.Version(*pypi_v.release)
    new_v = str(parsed_v.bump_prerelease())
    return new_v


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        raise ValueError("Usage: python test_version.py <package_name> <tag_version> [pypi_url]")

    package_name = sys.argv[1]
    tag_v = sys.argv[2]
    pypi_url = sys.argv[3] if len(sys.argv) == 4 else "https://pypi.org"

    latest_version, increment = get_latest_version(package_name, tag_v, pypi_url)
    if increment:
        new_version = increment_version(latest_version)
    else:
        new_version = latest_version

    # Output new version
    print(new_version)
