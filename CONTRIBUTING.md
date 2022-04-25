# Contributing Guide

Run all terminal commands in the root folder.

## Development

1. Create a fork/branch of main.

2. Setup the python environment using the provided PowerShell script.

    ```ps
    cicd/setup_env.ps1
    ```

3. Perform test driven development.

    Add or adjust tests in the tests folder and use these as the source of development.

4. Run tests.

    ``` cmd
    pytest
    ```

    Troubleshoot test collection if needed.

    ``` cmd
    pytest --collect-only
    ```

5. Set VERSION number in file `VERSION`.

    ```txt
    X.Y.Z
    ```

    X: major version (backwards incompatiable changes)
    Y: minor version (added backwards-compatible functionality)
    Z: patch version (bug fixes)

    Optionally a pre-release release candidate designation may be set in `VERSION`. This is useful for testing before final release.

    ```txt
    X.Y.ZrcN
    ```

    rc: specifies this is a release candidate
    N: release candidate number

6. Run `cicd/cicd_template.py`. If this finished to completion the CICD process will finish. Correct any errors as needed.

    ``` cmd
    python cicd/cicd_template.py
    ```

7. Create a Pull Request

## CICD Build Pipelines

Once a pull request is made it is time for the Continuous Integration / Continuous Delievery process. After CICD is completed a new version is uploaded to PyPI.

### Continuous Integration

A GitHub repository owner/contributor adds a comment of `/AzurePipelines run continuous-integration` on the pull request in GitHub. The continuous integration pipeline will run in Azure DevOps to ensure continous integration passes.

<!-- TODO: setup build pipeline -->
[Continuous Integration Pipeline](https://dev.azure.com/jasoncook1989/?/_build?definitionId=1)

### Continous Delivery

If the continuous integration pipeline passes, the continuous delievery pipeline will need to be manually started with an input for the build version number. After CD completes, a new version is uploaded to PyPI.

<!-- TODO: setup build pipeline -->
[Continuous Delivery Pipeline](https://dev.azure.com/jasoncook1989/?e/_build?definitionId=2)
