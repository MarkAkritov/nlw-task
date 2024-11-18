# api/Dockerfile
# Commands:
#    $ docker build -t api .
#    $ docker run -p 8000:8000 api

FROM python:3.11-buster

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y git

RUN apt-get install -y \
  build-essential \
  libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

ARG AZURE_OPENAI_API_VERSION
ENV AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION}

ARG AZURE_OPENAI_API_KEY
ENV AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}

ARG AZURE_OPENAI_ENDPOINT
ENV AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}

ARG AZURE_OPENAI_DEPLOYMENT
ENV AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT}

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

WORKDIR /api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "--app-dir=..", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
