# Docker Setup

This artifact is designed to run on CPUs and NVIDIA GPUs. To run on Intel or AMD GPUs or FPGAs, additional environment setup will be required. Please contact us for further guidance.

## Docker Environment Preparation

To prepare your environment for Docker, follow these steps:

1. **Remove Existing Docker Installation**

   If you have an older Docker installation, remove it using the following command:

   ```bash
   sudo apt-get remove docker docker-engine docker.io containerd runc
   ```

2. **Update Older Docker Versions**

   If you have Docker Community Edition (docker-ce) version 18.xx or older and have disabled updates, you can re-enable the repository by editing `/etc/apt/sources.list`. Locate the Docker repository entry, uncomment it if it is commented, or remove any "hold" applied using `sudo apt-mark unhold docker-ce`. After this, update your package list and install the updated Docker version. This can be done by removing a "hold" or uncommenting the relevant repository in `/etc/apt/sources.list`. The updated Docker version will upgrade to 19.03+ with standard updates.

## NVIDIA GPU Setup

Ensure that a recent NVIDIA driver is installed by following these steps:

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install build-essential dkms
sudo apt-get install nvidia-driver-560
```

The version installed above is "560". To find the latest driver version, press `[Tab]` twice after typing:

```bash
sudo apt-get install nvidia-driver-[Tab][Tab]
```

## Install Docker Community Edition (docker-ce)

Follow the steps below to install Docker CE:

1. Install required packages:

   ```bash
   sudo apt-get install \
       apt-transport-https \
       ca-certificates \
       curl \
       gnupg-agent \
       software-properties-common
   ```

2. Add and verify the Docker GPG key:

   ```bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   sudo apt-key fingerprint 0EBFCD88
   ```

   Ensure the fingerprint matches: `9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88`. If it does not match, stop the installation and verify the source of the GPG key. Check the official Docker documentation or reach out to support for assistance.

3. Add the Docker repository:

   ```bash
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   ```

4. Install Docker CE:

   ```bash
   sudo apt-get update
   sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

   Note: "containerd.io" is independent of Docker but is included in the same repository.

5. Verify the installation:

   ```bash
   sudo docker run --rm hello-world
   ```

   This command should pull and run the "hello-world" container from Docker Hub. At this point, only the `root` user can run Docker. Refer to the section on adding a user to the Docker group to enable non-root users to run Docker commands.

## Install NVIDIA Container Toolkit

Follow the NVIDIA documentation [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) to check the latest supported Ubuntu version. To install using `apt`:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

For installation on other distributions, refer to the NVIDIA documentation.

## User Group, User Namespace Configuration, and GPU Performance Tuning

Enhance Docker usability on a personal workstation:

1. **Add User to Docker Group**

   Add your user account to the Docker group to run Docker without `sudo`:

   ```bash
   sudo usermod -aG docker $USER
   ```

   Note: You need to log out and log back in for this to take effect.

2. **Restart Docker Service**

   Restart the Docker service to apply changes:

   ```bash
   sudo systemctl restart docker.service
   ```

Done! 

You have successfully set up Docker and configured your environment. To test your setup further, you can try running a containerized application, such as:

```bash
sudo docker run --rm -it ubuntu bash
```

This will pull and run the Ubuntu container interactively, allowing you to explore its functionality.

Additionally, to confirm NVIDIA GPU integration, you can run the `nvidia-smi` command via Docker:

```bash
sudo docker run --rm --gpus all -it ubuntu nvidia-smi
```

This command will display the GPU status and confirm that Docker can access the NVIDIA GPUs.