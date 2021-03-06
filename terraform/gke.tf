resource "google_container_cluster" "text_vec" {
  name     = "${var.cluster_name}"
  location = "${var.location}"

  remove_default_node_pool = true
  initial_node_count       = 1

  network = "${var.network}"

  min_master_version = "${var.min_master_version}"
  # remove_default_node_pool = trueの場合はnodeのversionを指定できない
  # node_version       = "${var.node_version}"

  master_auth {
    # https://github.com/hashicorp/terraform-provider-google/issues/8032
    # basic認証はdeprecated
    # username = ""
    # password = ""

    client_certificate_config {
      issue_client_certificate = false
    }
  }
}

resource "google_container_node_pool" "text_vec_nodes" {
  name       = "${var.cluster_name}-nodes"
  location   = "${var.location}"
  cluster    = "${google_container_cluster.primary.name}"
  node_count = "${var.primary_node_count}"

  management {
    auto_repair = true
  }

  node_config {
    oauth_scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append",
    ]

    machine_type = "${var.machine_type}"

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
}