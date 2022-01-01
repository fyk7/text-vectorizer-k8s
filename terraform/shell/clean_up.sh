# declare variables
GCP_PROJECT=$(gcloud info --format='value(config.project)')
TERRAFORM_SA=terraform-service-account
GCS_BUCKET=text-vect-terraform
TERRAFORM_SA_EMAIL=$(gcloud iam service-accounts list \
  --project=$GCP_PROJECT \
  --filter="displayName:$TERRAFORM_SA" \
  --format='value(email)')


# remove gcs bucket
gsutil rm -r gs://$GCS_BUCKET/

# remove iam policies
gcloud projects remove-iam-policy-binding $GCP_PROJECT \
  --role roles/iam.serviceAccountUser \
  --member serviceAccount:$TERRAFORM_SA_EMAIL

gcloud projects remove-iam-policy-binding $GCP_PROJECT \
  --role roles/compute.admin \
  --member serviceAccount:$TERRAFORM_SA_EMAIL

gcloud projects remove-iam-policy-binding $GCP_PROJECT \
  --role roles/storage.admin \
  --member serviceAccount:$TERRAFORM_SA_EMAIL

gcloud projects remove-iam-policy-binding $GCP_PROJECT \
  --role roles/container.clusterAdmin \
  --member serviceAccount:$TERRAFORM_SA_EMAIL

# remove service account
gcloud iam service-accounts delete $TERRAFORM_SA_EMAIL
