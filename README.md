# Gene circuit evolutionary stability

Build, simulate, and analyse genetic circuits and the effect of mutations.

## Usage

### Docker

Should take ca. 1 minute to load and start container. If you want to start the docker container, follow these steps.

_Linux_
1. From your terminal, change directories to be inside the home directory of this project. For example you can do this by using the `cd` and navigating to the `synbio_morpher` directory.
2. Run the setup/start_docker.sh using bash, eg `bash setup/startup_docker.sh`. If you run this from another directory, the docker container will simply contain all the folder in the directory you ran this command from rather than only containing the folders within the scope of this project.


_Windows_
While running docker from Windows is very similar to Linux as the current version of Docker Desktop ships with WSL (a linux emulator), Windows-specific powershell commands such as pathnames, variable definition and line breaks have been corrected for in the `.cmd` version of `startup_docker`.
1. From your power shell, change directories to be inside the home directory of this project. For example you can do this by using the `cd` and navigating to the `synbio_morpher` directory.
2. Run the setup/start_docker.cmd, eg `.\setup\startup_docker.cmd`. If you run this from another directory, the relative pathnames specified in the cmd script to the `docker` directory in the project will not work.

Some important things to keep in mind for editing the windows Dockerfile or `.cmd` file - within the container, docker uses the linux file system convention, so use forward slashes '/' as you would in a UNIX environment for any file references that are inside the container. Any references to local Windows files should use backslashes in the `.cmd` file but forward slashes in the Dockerfile.
