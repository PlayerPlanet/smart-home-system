FROM python:3.10-alpine
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    linux-headers \
    libffi-dev \
    py3-numpy \
    build-base \
    lapack-dev \
    blas-dev

WORKDIR /code
ENV FastAPI_HOST=0.0.0.0
ENV PORT=5000
ENV TEST=true
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
EXPOSE 5000
COPY . .
CMD uvicorn src.app:app --host $FastAPI_HOST --port $PORT