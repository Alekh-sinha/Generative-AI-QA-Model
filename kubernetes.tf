provider "google" {
  project = "qualified-abode-411820"
  region  = "us-central1"
}

resource "google_container_cluster" "primary" {
  name     = "upsc-2024-cluster"
  location = "us-central1-a"

  node_pool {
    name            = "default-pool"
    node_count      = 1
    autoscaling {
      min_node_count = 0
      max_node_count = 1
    }
    node_config {
      machine_type = "n1-highmem-8"
      disk_size_gb = 50
    }
  }
}

resource "kubernetes_deployment" "upsc-2024" {
  metadata {
    name = "upsc-2024-deployment"
    labels = {
      app = "upsc-2024"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "upsc-2024"
      }
    }

    template {
      metadata {
        labels = {
          app = "upsc-2024"
        }
      }

      spec {
        container {
          image = "gcr.io/qualified-abode-411820/pytorch_predict_upsc_2014:latest"
          name  = "upsc_2014"
          image_pull_policy = "Always" # Set image pull policy to "Always"
        }
      }
    }
  }
}

resource "kubernetes_service" "upsc-2024" {
  metadata {
    name = "upsc-2024-service"
    labels = {
      app = "upsc-2024"
    }
  }

  spec {
    selector = {
      app = "upsc-2024"
    }

    port {
      protocol = "TCP"
      port     = 80
      target_port = 7860
    }
  }
}

resource "kubernetes_ingress" "upsc-2024" {
  metadata {
    name = "upsc-2024-ingress"
    annotations = {
      "nginx.ingress.kubernetes.io/proxy-connect-timeout" = "900s"
      "nginx.ingress.kubernetes.io/proxy-send-timeout"    = "900s"
      "nginx.ingress.kubernetes.io/proxy-read-timeout"    = "900s"
    }
  }

  spec {
    rule {
      http {
        path {
          backend {
            service_name = kubernetes_service.upsc-2024.metadata[0].name
            service_port = kubernetes_service.upsc-2024.spec[0].port[0].port
          }
        }
      }
    }
  }
}
