FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    numpy==2.4.4 \
    scipy==1.16.3

WORKDIR /root

COPY data /root/data

RUN mkdir -p /root/results
