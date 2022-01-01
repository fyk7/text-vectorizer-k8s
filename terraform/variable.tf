variable "project" {}

variable "cluster_name" {
  default = "text-vec-cluster"
}

variable "location" {
  default = "asia-northeast1-a"
}

# 実運用では明示的に作成したnetwork(defauld以外)を指定した方が良さげ?
variable "network" {
  default = "default"
}

variable "primary_node_count" {
  default = "3"
}

variable "machine_type" {
  default = "n1-standard-1"
}

variable "min_master_version" {
  default = "1.20.11-gke.1300"
}

variable "node_version" {
  default = "1.20.11-gke.1300"
}