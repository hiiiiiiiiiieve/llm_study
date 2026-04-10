FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY . .

CMD ["python", "hello.py"]