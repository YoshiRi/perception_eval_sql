# syntax=docker/dockerfile:1

# Evaluation Dashboard (Streamlit)
# See Readme.md "Docker" section for usage/build tips.
# Build example: docker build --secret id=ssh,src=$HOME/.ssh/id_rsa -t evaluation-dashboard .
# Match ROS distro at build: --build-arg ROS_DISTRO=humble (or iron, jazzy, etc.)
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}

WORKDIR /app

# Make ROS_DISTRO available inside the container at runtime (for entrypoint sourcing)
ENV ROS_DISTRO=${ROS_DISTRO}

# Install needed system packages: python, geos, git/ssh (for pip install from private repos), pipx, OpenGL for matplotlib
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      python3-pip python3-venv libgeos-c1v5 bash git openssh-client libgl1-mesa-glx libgl1-mesa-dri \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip, install pipx, ensure pipx path available
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging pipx \
 && pipx ensurepath
ENV PATH="/root/.local/bin:${PATH}"

RUN git config --global url."ssh://git@github.com/".insteadOf "https://github.com/"
# Install private deps using SSH secret, combine to minimize layers, avoid repetition
RUN --mount=type=secret,id=ssh,dst=/tmp/ssh_key \
    mkdir -p /root/.ssh \
 && cp /tmp/ssh_key /root/.ssh/id_rsa \
 && chmod 700 /root/.ssh \
 && chmod 600 /root/.ssh/id_rsa \
 && export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_rsa -o StrictHostKeyChecking=accept-new" \
 && python3 -m pip install --no-cache-dir "git+ssh://git@github.com/tier4/webauto-auth-py.git" \
 && pipx install "git+ssh://git@github.com/tier4/v_and_v_util.git" \
 && python3 -m pip install --no-cache-dir "git+ssh://git@github.com/tier4/autoware_perception_evaluation.git" \
 && python3 -m pip install --no-cache-dir "git+ssh://git@github.com/tier4/perception_catalog_analyzer.git" \
 && rm -rf /root/.ssh # Clean up private key ASAP for security

# Install public dependencies (after SSH deps so SSH failures surface fast)
COPY requirements-docker.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-docker.txt

# Streamlit server options (WebSocket proxy, headless, etc.)
COPY .streamlit/ .streamlit/

# Copy application code and config
COPY Overview.py .
COPY pages/ pages/
COPY lib/ lib/
COPY worker/ worker/
COPY configs/ configs/
COPY docs/ docs/
COPY docker-entrypoint.sh .

RUN chmod +x /app/docker-entrypoint.sh

# Create data folder (should be mounted in production)
RUN mkdir -p /app/data

EXPOSE 8501

# Entrypoint runs script that sources ROS + optional pilot-auto, then launches Streamlit
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
