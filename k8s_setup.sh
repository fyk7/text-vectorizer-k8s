# fastapi namespace

{
  kubectl create ns fastapi-textvec
} || {
  echo 'fastapi-textvec namespace is already defined'
}

kubectl apply -n fastapi-textvec -f k8s/fastapi/deployment.yaml

kubectl apply -n fastapi-textvec -f k8s/fastapi/service.yaml 

kubectl get -n fastapi-textvec pods

kubectl get -n fastapi-textvec svc


# celery-textvec namespace

{
  kubectl create ns celery-textvec
} || {
  echo 'celery-textvec namespace is already defined'
}

kubectl apply -n celery-textvec -f k8s/celery/worker-deployment.yaml

kubectl get -n celery-textvec pods


# redis-textvec namespace

{
  kubectl create ns redis-textvec
} || {
  echo 'redis-textvec namespace is already defined'
}

kubectl apply -n redis-textvec -f k8s/redis/deployment.yaml

kubectl apply -n redis-textvec -f k8s/redis/service.yaml

kubectl get -n redis-textvec pods

kubectl get -n redis-textvec svc
