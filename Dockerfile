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
ENV FastAPI_APP=app.py
ENV FastAPI_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt
EXPOSE 5000
COPY . .
CMD ["fastapi", "run", "app.py", "--port", "5000"]