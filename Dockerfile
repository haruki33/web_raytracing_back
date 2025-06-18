FROM python:3.12-slim

ENV TZ Asia/Tokyo

WORKDIR /workspace/backend

RUN apt update && \
    apt -y upgrade

COPY requirements.txt /workspace/backend
RUN pip install --upgrade pip && pip install -r ./requirements.txt

EXPOSE 8000
