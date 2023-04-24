
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
sudo docker run --name inference_mnist_api -d -p 8888:80 develop/mnist_inference
```

  - Inference for port: 8888 docker container
```bash=
image_b64_3=$(base64 ./3.jpg | tr -d '\n');echo $image_b64_3
curl -X POST -H "Content-Type: application/json" -d '{"image": "'$image_b64_3'"}' http://localhost:8888/predict
```

![](https://imgur.com/Vtx7giN.png)

  - push image to docker hub

```
sudo sudo docker tag 9fc4e7c2f061 ponddy/mnist_inference:v1

# push to mnist_inference:v1 in docker hub
sudo sudo docker tag 9fc4e7c2f061 ponddy/mnist_inference:v1
sudo docker push ponddy/mnist_inference:v1
```

  - K8S service / deployment / ingress setting
    - service / deplpyment : ==mnist-prd.yaml==

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-mnist
  namespace: ml-prd
  labels:
    app: ml-mnist
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-mnist
  template:
    metadata:
      labels:
        app: ml-mnist
    spec:
      containers:
        - image: ponddy/mnist_inference:v1
          resources:
            requests:
              memory: "100Mi"
              # cpu: "1000m"
          name: ml-mnist
      imagePullSecrets:
        - name: regcred
---
apiVersion: v1
kind: Service
metadata:
  namespace: ml-prd
  name: ml-mnist
  labels:
    app: ml-mnist
    service: ml-mnist
spec:
  ports:
  - port: 80
    targetPort: 80
    name: http
  selector:
    app: ml-mnist

```

  - apply mnist-prd.yaml
```bash=
kubectl apply -f mnist-prd.yaml
```

  - ingress: ingress_ML-prd.yaml
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-prd-ingress
  namespace: ml-prd
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/proxy-body-size: "7m"
    #ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rewrite-target: /$2
    nginx.ingress.kubernetes.io/use-regex: "true"    
spec:
  rules:
  # ml-prd.ponddy.org & ml-dev.ponddy.com 都要設定  
  - host: ml-prd.k8s.wenwen999.link
    http:
      paths:
      - path: /mnist(/|$)(.*)
        pathType: Prefix
        backend:
          service:
            name: ml-mnist
            port:
              name: http                 
```
  - apply ingress_ML-prd.yaml
```bash=
kubectl apply -f ingress_ML-prd.yaml
```

   - Views pod / service / deployment / ingress
```bash=
echo "========================ml-prd==============================="
echo "------------------------------------------------------------"
echo "              pod"
echo "------------------------------------------------------------"
kubectl get pods -n ml-prd -o wide
echo "------------------------------------------------------------"
echo "              deployments"
echo "------------------------------------------------------------"
kubectl get deployments -n ml-prd -o wide
echo "------------------------------------------------------------"
echo "              autoscaling status"
echo "------------------------------------------------------------"
kubectl get hpa -n ml-prd
echo "------------------------------------------------------------"
echo "              ingress description"
echo "------------------------------------------------------------"
kubectl describe ingress ml-prd-ingress -n ml-prd
echo "------------------------------------------------------------"
```

![](https://imgur.com/MpyhyGC.png)

  - ==Call k8s mnist example== 
     - Endpoint: https://ml-prd.k8s.wenwen999.link/mnist/predict
```
image_b64_3=$(base64 ./3.jpg | tr -d '\n');echo $image_b64_3
curl -X POST -H "Content-Type: application/json" -d '{"image": "'$image_b64_3'"}' https://ml-prd.k8s.wenwen999.link/mnist/predict
```
![](https://imgur.com/aok62qy.png)

