terraform {
  backend "gcs" {
    bucket = "text-vect-terraform-v2"
    prefix = "terraform/state"
  }
}