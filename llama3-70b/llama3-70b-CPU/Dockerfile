FROM python:3.9

WORKDIR /app

COPY app2.py /app/app2.py

RUN pip install accelerate
RUN pip install torch torchvision torchaudio transformers

CMD ["python", "app2.py"]

