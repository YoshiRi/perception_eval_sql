# syntax=docker/dockerfile:1

# Evaluation Dashboard (Streamlit)
# See Readme.md "Docker" section for usage/build tips.
# Build example: docker build --secret id=ssh,src=$HOME/.ssh/id_rsa -t evaluation-dashboard .
# Match ROS distro at build: --build-arg ROS_DISTRO=humble (or iron, jazzy, etc.)
ARG ROS_DISTRO=humble
ARG WEBAUTOAUTHCLI_COMMIT=204629123fa58ab0be0966c795703324e34851ec
FROM ros:${ROS_DISTRO}

WORKDIR /app

# Make ROS_DISTRO available inside the container at runtime (for entrypoint sourcing)
ENV ROS_DISTRO=${ROS_DISTRO}

# Install needed system packages: python, geos, git/ssh (for pip install from private repos), pipx,
# OpenGL for matplotlib, and Chrome for Kaleido static image export.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      python3-pip python3-venv libgeos-c1v5 bash git openssh-client \
      libgl1-mesa-glx libgl1-mesa-dri ca-certificates curl gnupg \
      fontconfig fonts-noto-cjk fonts-ipafont-gothic \
 && install -d -m 0755 /etc/apt/keyrings \
 && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
    | gpg --dearmor -o /etc/apt/keyrings/google-chrome.gpg \
 && chmod a+r /etc/apt/keyrings/google-chrome.gpg \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] https://dl.google.com/linux/chrome/deb/ stable main" \
    > /etc/apt/sources.list.d/google-chrome.list \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends google-chrome-stable \
 && ln -sf /usr/bin/google-chrome-stable /usr/bin/google-chrome \
 && fc-cache -f \
 && rm -rf /var/lib/apt/lists/*

ENV CHROME_BIN=/usr/bin/google-chrome-stable

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
 && python3 -m pip install --no-cache-dir "git+ssh://git@github.com/tier4/WebAutoAuthCLI.git@v2.23.1" \
 && python3 -m pip install --no-cache-dir "git+ssh://git@github.com/tier4/webauto-auth-py.git" \
 && pipx install "git+ssh://git@github.com/tier4/v_and_v_util.git" \
 && python3 -m pip install --no-cache-dir "git+ssh://git@github.com/tier4/autoware_perception_evaluation.git" \
 && python3 -m pip install --no-cache-dir \
      "bokeh>=3.7.3" \
      "lz4>=4.4.4" \
      "pyarrow==19.0.0" \
      "tabulate>=0.9.0" \
      "typer>=0.16.0" \
      "weasyprint>=65.1" \
 && git clone --depth 1 git@github.com:tier4/perception_catalog_analyzer.git /opt/perception_catalog_analyzer \
 && python3 -m pip install --no-cache-dir --no-deps -e /opt/perception_catalog_analyzer \
 && rm -rf /root/.ssh # Clean up private key ASAP for security

# Install public dependencies (after SSH deps so SSH failures surface fast)
COPY requirements-docker.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-docker.txt


# Copy application code and config
COPY Overview.py .
COPY Readme.md .
COPY Readme.en.md .
COPY catalogs.json .
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
