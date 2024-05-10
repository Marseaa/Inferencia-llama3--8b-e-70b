
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y wget 

# instalando python
RUN apt-get install -y python3  && apt-get install -y python3-pip

WORKDIR /app

#COPY Meta-Llama-3-8B-Instruct /app/Meta-Llama-3-8B-Instruct
#COPY app.py /app/app.py

RUN pip install accelerate
RUN pip install torch torchvision torchaudio transformers

ENV NVIDIA_VISIBLE_DEVICES all

CMD ["python3", "app.py"]

