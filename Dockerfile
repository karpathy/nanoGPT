FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update \
    && apt-get install -y \
    && apt-get -y install gcc 
   
CMD ["/bin/bash"]