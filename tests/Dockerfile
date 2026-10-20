FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

# The verifier runs the submitted solver with the same scientific runtime the agent was given.
RUN pip install --no-cache-dir \
    numpy==2.4.4 \
    scipy==1.16.3 \
    pytest==8.4.1 \
    pytest-json-ctrf==0.3.5

RUN mkdir -p /app /root/results /logs/verifier \
    && useradd --system --uid 65532 --no-create-home --shell /usr/sbin/nologin solveruser

# Private campaigns and their ground truth ship inside this image, not on any public host. The agent container has network
# access, so anything reachable by URL would be reachable by a submission.
COPY . /tests/

# Pytest stays root while the submitted solver is demoted to solveruser. The demoted account receives only staged inputs under
# /app and cannot traverse verifier code, private campaigns or ground truth under /tests, nor read or write /logs/verifier.
RUN chmod 700 /tests /tests/test.sh /logs/verifier

WORKDIR /tests
