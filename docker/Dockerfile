# set ubuntu as backend OS
FROM python:3.10.13

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

# Install pip packages.
RUN pip -q install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    

COPY . /app/pythia

RUN pip -q install --no-cache-dir  -r /app/pythia/requirements.txt

WORKDIR /app/pythia

RUN echo $'#!/bin/bash\n\
ldconfig\n\
python3 /app/pythia/pythia/masked_ddg_scan.py "$@"' > /app/run_pythia.sh \
  && chmod +x /app/run_pythia.sh
ENTRYPOINT ["/app/run_pythia.sh"]