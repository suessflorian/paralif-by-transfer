provider "google" {
  project     = "graphic-armor-339522"
  region      = "asia-northeast1-b"
}

resource "google_storage_bucket" "checkpoints" {
  name          = "florians_results"
  location      = "asia-northeast1"
  force_destroy = true  # Forces bucket deletion even if it contains objects
}

resource "google_artifact_registry_repository" "docker_registry" {
  provider = google
  location = "asia-northeast1"
  repository_id = "florians-trainer"
  format = "DOCKER"
  description = "Docker registry for florians-trainer"
}

output "docker_registry_url" {
  value = "asia-northeast1-docker.pkg.dev/${google_artifact_registry_repository.docker_registry.project}/${google_artifact_registry_repository.docker_registry.repository_id}"
}
