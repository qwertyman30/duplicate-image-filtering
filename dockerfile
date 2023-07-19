FROM python:3

# Needed for opencv
RUN apt-get update && apt-get install -y libgl1

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]