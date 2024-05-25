provider "google" {
  project     = "graphic-armor-339522"
  region      = "asia-northeast1-b"
}

resource "google_storage_bucket" "checkpoints" {
  name          = "florians_checkpoints"
  location      = "ASIA-NORTHEAST1"
  force_destroy = true  # Forces bucket deletion even if it contains objects
}
