FROM python:3.8-bullseye

WORKDIR /app

EXPOSE 80

RUN apt-get update && apt-get install -y logrotate

RUN python3 -m pip install --upgrade pip

COPY ./requirements.txt .

RUN pip install -r requirements.txt

# 这里将flask安装移到一个独立的RUN命令中
RUN pip install flask

COPY . .

ENV FLASK_APP=inference.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]

