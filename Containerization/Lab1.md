**Not really for container**
## Python Environment Manager
1. `conda create -n ai python=3.10 -y (-y for auto consent)`
  - Tells Conda to build a new, isolated "room" (environment) with the name "ai" for your project
2. `conda info --envs`
  - List all known conda environments and their file system location
3. `conda activate ai`
  - Any libraries you install now (like NumPy or PyTorch) will stay inside this environment(ai) and won't clutter up your main computer
4. `python -V`
  - To list the version for the Python versions(should be 3.10 for this)
5. `conda deactivate`
  - To deactivate an active environment

# Docker Engine Install

## Phase 1: Security Check
1. `sudo install -m 0755 -d /etc/apt/keyrings`
  - Creates a specific folder (keyrings) to store security keys. The `-m 0755` sets the permissions so the system can read it, but only the admin can change it
2. `curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg`
  - `curl`: downloads Docker’s "digital signature (GPG key)
  - `gpg --dearmor`: converts that key into a format Linux understands and saves it as `docker.gpg`

## Phase 2: Tell Linux where to Look
3. 
  ```
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  ```
  - This adds Docker's official "store address" (repository) to your system's list of sources
  - Uses `dpkg --print-architecture` to automatically figure out if you're on a standard PC (amd64) or an ARM chip (like a Raspberry Pi or AWS Graviton)

## Phase 3: Installation
4. `sudo apt update`
  - Refreshes your local "catalog" of software. Now that you added Docker's address in the previous step, your computer "sees" Docker for the first time
5. `sudo apt -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`
  - `docker -ce`: Community Edition Engine
  - docker-ce-cli`: command-line interface (`docker` command)
  - `containered.io`: background service that manages the life cycle of containers
  - `docker-compose-plugin`: A tool to run multiple containers at once

## Phase 4:
6. `sudo usermod -aG docker $USER`
  - By default, only "Root" (the superuser) can run Docker. This adds you ($USER) to the Docker group so you don't have to type sudo every single time you run a container
  - If you don't run this command, you have to type `sudo docker ...` for every single action
  - `-aG`: Append to a Group. If forgot the `-a`, you would remove yourself from all other groups (Ex: the one that lets you use sudo or internet)
7. `newgrp docker`
  - Linux usually requires you to log out and back in for group changes to take effect. This forces the current terminal window to re-read its permission list right now. 
    - Saves you from having to close all your apps and log out of Ubuntu just to start working
  - **Caution**: While newgrp docker works for your current terminal window, any other terminals you already have open won't know about the change
    - **Security Tradeoff**:
      - By adding yourself to `docker` group, you are giving your user account a lot of power. Anyone who can run a Docker command can gain "root" access to your host files if they know what they are doing
8. `docker run hello-world`  
  - Runs a full diagnostic test of the four main pillars of Docker
  1. `The pull`: Docker talk to internet (Docker) to find an image
  2. `The storage`: Docker save that image to your hard drive 
  3. `The isolation`: Docker create a "container"
  4. `The execution`: The code inside that container actually run and send text back to your screen

## Clean up after `hello-world` Test
  - To truly clean up, you have to throw away the container first, then delete the recipe
1. **Find the ID**: `docker ps -a`
  - Look for entry where IMAGE is `hello-world`
2. **Remove it**: `docker rm <container_id>`
  - **Shortcut**: `docker container prune` - Remove all stopped container at once
3. **List image**: `docker images`
4. **Remove the image**: `docker rmi hello-world:<version>`