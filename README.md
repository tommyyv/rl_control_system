# MACHINE LEARNING ASSISTED CONTROL SYSTEM

## Introduction
This project demonstrates a reinforcement learning algorithm implementation in a
control system. A baseline and a hybrid was conducted to compare the difference
between a classical control method versus an optimized approach.

## Install Guide
This section includes a guide for users who plan on forking this repository and running the Docker image using Docker Desktop. Docker Desktop promotes portability and flexibility across platforms; Windows, macOS, and/or Linux users have access to the source code as long as the Dock Desktop application is available to install on their machine. The Dockerfile runs a light-weight Debian instance, as a base image and executes the following commands to build the source code and start the application. Below are instructions on how a user would install and use the application within a Docker instance.

**NOTE**: Building the Docker image and running the Python application will require extensive time to complete.

**NOTE**: You may need to elevate to root privileges execute commands. This command sequence is based on Linux.

1. Open a terminal or command prompt instance (recommend using PowerShell for cross-platform alias commands)
2. Clone the Git repository
**NOTE**: This will clone the Git repository in your current working directory.
`git clone https://github.com/tommyyv/rl_control_system.git`
3. Navigate to the project’s root directory, if you have not already
`cd rl_control_system`
4. Verify a Dockerfile exists in the project’s root directory
`ls -lah`
5. Build the Docker image
`sudo docker build -t rl_control_system .`
6. Run Docker container
(macOS/Linux): `sudo docker run -it --rm rl_control_system python -m main`

## User Guide
1. Navigate to the interactive Google Colab URL:
2. Upload the provided .ipynb file
3. Create a results directory within the Google Colab workspace
4. Upload the CSV files within the results directory
**NOTE**: There may be multiple directories within the results directory due to multiple program runs. Open the desired directory within the results directory and upload the (2) results CSV files. e.g. YYYYMMDD_THH:MM_results contains (2) CSV files, (1) error_metrics.txt, and (2) PNG files.
5. Execute the notebook by selecting the Run All button found in the navigation bar
