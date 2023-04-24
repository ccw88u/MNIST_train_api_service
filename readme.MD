
## install
 
```
python3.8 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


## Train model
   - 80% for train and vaildation
   - callbacks
     - early_stop: val_loss patience=3
     - checkpoint: save val_loss save_best_only: model name:==best_model.h5==
```bash=
python3 train.py
```

![](https://imgur.com/Z8U10yL.png)

## flask api and inference
  - Run flask
```bash=
python3 inference.py
```
  - Run local curl to print inference result
```bash=
image_b64_3=$(base64 ./3.jpg | tr -d '\n');echo $image_b64_3
curl -X POST -H "Content-Type: application/json" -d '{"image": "'$image_b64_3'"}' http://localhost:5000/predict

image_b64_4=$(base64 ./4.jpg | tr -d '\n');echo $image_b64_4
curl -X POST -H "Content-Type: application/json" -d '{"image": "'$image_b64_4'"}' http://localhost:5000/predict
```
![](https://imgur.com/8vdYj0N.png)


## Build Docker image and deploy to k8s 
  - Docker file
```
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
```
  - Build docker image
```bash=
sudo docker build --no-cache -t "develop/mnist_inference" -f ./Dockerfile .
```
  - Run in test docker container
```bash=
sudo docker run --name inference_mnist_api -d -p 8888:80 evelop/mnist_inference
```
  - Inference for port: 8888

```bash=
image_b64_3=$(base64 ./3.jpg | tr -d '\n');echo $image_b64_3
curl -X POST -H "Content-Type: application/json" -d '{"image": "'$image_b64_3'"}' http://localhost:8888/predict
```
![](https://imgur.com/Vtx7giN.png)

