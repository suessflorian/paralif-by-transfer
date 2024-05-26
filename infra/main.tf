provider "google" {
  project     = "graphic-armor-339522"
  region      = "asia-northeast1-b"
}

resource "google_storage_bucket" "checkpoints" {
  name          = "florians_results"
  location      = "asia-northeast1"
  force_destroy = true  # Forces bucket deletion even if it contains objects
}

# NOTE: we will revisit this, for now we will just use a manual ssh/scp file process
# resource "google_artifact_registry_repository" "docker_registry" {
#   provider = google
#   location = "asia-northeast1"
#   repository_id = "florians-trainer"
#   format = "DOCKER"
#   description = "Docker registry for florians-trainer"
# }
#
# output "docker_registry_url" {
#   value = "asia-northeast1-docker.pkg.dev/${google_artifact_registry_repository.docker_registry.project}/${google_artifact_registry_repository.docker_registry.repository_id}"
# }

resource "google_compute_instance" "florians-deeplearning" {
  boot_disk {
    auto_delete = true
    device_name = "florians-deeplearning"

    initialize_params {
      image = "projects/debian-cloud/global/images/debian-12-bookworm-v20240515"
      size  = 10
      type  = "pd-balanced"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  labels = {
    goog-ec-src = "vm_add-tf"
  }

  machine_type = "e2-medium"
  name         = "florians-deeplearning"

  network_interface {
    access_config {
      network_tier = "PREMIUM"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = "projects/graphic-armor-339522/regions/us-central1/subnetworks/default"
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  service_account {
    email  = "619899876982-compute@developer.gserviceaccount.com"
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  zone = "us-central1-a"
}
