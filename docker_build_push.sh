# kubernetesで使用するためのimageを一括でbuildするためにdocker-composeを利用する
docker-compose -f docker-compose-k8s.yml build

# gcloud auth configure-docker
# GCPコンソールでgoogle container registoryのAPIを有効にする
# gcloud config set <project_name>

GCP_PROJECT=$(gcloud info --format='value(config.project)')

docker tag fyk7/celery-fastapi-text-vect-api gcr.io/$GCP_PROJECT/fyk7/celery-fastapi-text-vect-api:latest
docker tag fyk7/celery-fastapi-text-vect-worker gcr.io/$GCP_PROJECT/fyk7/celery-fastapi-text-vect-worker:latest

docker push gcr.io/$GCP_PROJECT/fyk7/celery-fastapi-text-vect-api:latest
docker push gcr.io/$GCP_PROJECT/fyk7/celery-fastapi-text-vect-worker:latest