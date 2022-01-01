# kubernetesで使用するためのimageを一括でbuildするためにdocker-composeを利用する
docker-compose -f docker-compose-k8s.yml build

# gcloud auth configure-docker
# gcloud config set <project_name>

docker tag fyk7/celery-fastapi-text-vect-api gcr.io/text-vectorize/fyk7/celery-fastapi-text-vect-api:latest
docker tag fyk7/celery-fastapi-text-vect-worker gcr.io/text-vectorize/fyk7/celery-fastapi-text-vect-worker:latest

docker push gcr.io/text-vectorize/fyk7/celery-fastapi-text-vect-api:latest
docker push gcr.io/text-vectorize/fyk7/celery-fastapi-text-vect-worker:latest