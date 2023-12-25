# set ubuntu as backend OS
FROM ubuntu:24.04

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qy  \
    && apt-get install -qy ca-certificates \
    && apt-get install -qy python3 python3-dev python3-venv python3-pip perl wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/pythia

# Install pip packages.
RUN pip3 -q install --no-cache-dir --upgrade pip \
    && pip3 -q install --no-cache-dir  -r /app/pythia/requirements.txt \
    && pip3 -q install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    


# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

WORKDIR /app/pythia

RUN echo $'#!/bin/bash\n\
ldconfig\n\
python3 /app/pythia/pythia/masked_ddg_scan.py "$@"' > /app/run_pythia.sh \
  && chmod +x /app/run_pythia.sh
ENTRYPOINT ["/app/run_pythia.sh"]