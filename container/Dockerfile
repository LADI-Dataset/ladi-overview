FROM ubuntu

# part 1: install python, pip, relevant libraries
SHELL ["/bin/bash", "-c"]
RUN apt update && apt install -y python3 python3-setuptools python3-pip python3-venv ffmpeg libsm6 libxext6
RUN python3 -m venv /cap_env && source /cap_env/bin/activate && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && pip3 install boto3 opencv-python numpy transformers tqdm pandas

# part 2: get code necessary to download base models and run inference
WORKDIR /app
COPY . .
WORKDIR /app/inference
# download models from docker hub, then run a test infer to trigger
RUN source /cap_env/bin/activate && python dl_models.py

ENTRYPOINT /bin/bash