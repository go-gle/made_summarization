FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /code
COPY requirements.txt /code/
COPY src/ /code/
RUN pip3 install -r requirements.txt

CMD python3 main.py
