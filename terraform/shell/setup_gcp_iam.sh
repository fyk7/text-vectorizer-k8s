# gcloud services enable container.googleapis.com
# brew install terraform

# 1行ずつ実行する
GCP_PROJECT=$(gcloud info --format='value(config.project)')
TERRAFORM_SA=terraform-service-account-v2
gcloud iam service-accounts create $TERRAFORM_SA --project=$GCP_PROJECT --display-name $TERRAFORM_SA

TERRAFORM_SA_EMAIL=$(gcloud iam service-accounts list \
  --project=$GCP_PROJECT \
  --filter="displayName:$TERRAFORM_SA" \
  --format='value(email)')


gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --role roles/iam.serviceAccountUser \
  --member serviceAccount:$TERRAFORM_SA_EMAIL

gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --role roles/compute.admin \
  --member serviceAccount:$TERRAFORM_SA_EMAIL

gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --role roles/storage.admin \
  --member serviceAccount:$TERRAFORM_SA_EMAIL

gcloud projects add-iam-policy-binding $GCP_PROJECT \
  --role roles/container.clusterAdmin \
  --member serviceAccount:$TERRAFORM_SA_EMAIL



# create google cloud storage bucket for terraform .tfstate
GCS_CLASS=multi_regional
GCS_BUCKET=text-vect-terraform-v2
gsutil mb -p $GCP_PROJECT -c $GCS_CLASS -l asia gs://$GCS_BUCKET/



TERRAFORM_SA_DEST=$HOME/.gcp/terraform-service-account.json
mkdir -p $(dirname $TERRAFORM_SA_DEST)
TERRAFORM_SA_EMAIL=$(gcloud iam service-accounts list \
  --filter="displayName:$TERRAFORM_SA" \
  --format='value(email)')
gcloud iam service-accounts keys create $TERRAFORM_SA_DEST \
  --iam-account $TERRAFORM_SA_EMAIL

export GOOGLE_APPLICATION_CREDENTIALS=$TERRAFORM_SA_DEST

# terraformのvariable.tfファイルにおいてdefault値を記載していないvariableに対して値を設定する
export TF_VAR_project=$GCP_PROJECT
