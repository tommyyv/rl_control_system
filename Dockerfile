FROM python:3.11-slim-buster

WORKDIR /rl_control_system

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ['python', '-m', 'main']
