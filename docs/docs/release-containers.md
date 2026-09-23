# CDS 1.2.0 container images

The release distributes two Linux x86-64 service images through NGC. GitHub hosts the source code, not the container images.

| Service | Image | Manifest digest |
| --- | --- | --- |
| CDS API | `nvcr.io/nvidia/blueprint/cosmos-dataset-search:1.2.0` | `sha256:e591652c39684cce9c6409af1cd361ec03d3bb9aa3397c3924401bfe4e5b8c2e` |
| CE1 OSS | `nvcr.io/nvidia/blueprint/cosmos-embed1-oss:1.2.0` | `sha256:53bb78a6813bdfd1e07fa97ca63ff822bec9fb226a11fb54aee81f47f16b2dec` |

CE1 requires an NVIDIA GPU with NVDEC support. Model weights are provisioned separately; they are not included in these images. Third-party dependencies and model weights retain their respective licenses.

## Use prebuilt images with Docker Compose

Authenticate to NGC with a key that can access both containers. Keep the key out of source files and command-line arguments:

```bash
printf '%s' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Follow the [deployment prerequisites](docker-compose-prerequisites.md), prepare the pinned model with `make download-ce1-model`, and configure `deploy/standalone/.env` as described in the [Compose guide](docker-compose-deployment.md). Set these two overrides in that file:

```dotenv
VISUAL_SEARCH_IMAGE=nvcr.io/nvidia/blueprint/cosmos-dataset-search:1.2.0
COSMOS_EMBED_IMAGE=nvcr.io/nvidia/blueprint/cosmos-embed1-oss:1.2.0
```

From `deploy/standalone`, start the existing stack without rebuilding:

```bash
docker compose -f docker-compose.build.yml up -d --no-build --pull always
```

To pin the pulled image, replace each `:1.2.0` suffix with `@sha256:...` using the digest in the table. This development Compose file also bind-mounts local CDS/Haystack source over the CDS image; keep the checkout at the release version, or remove those source mounts when you want to run only the packaged code. The source-build workflow remains available with `make build-docker` and the local image defaults.

## Kubernetes

The Helm source defaults to the same `1.2.0` images. Provision the model PVC and follow the [AWS EKS guide](aws-eks-deployment.md). Image repositories and tags remain configurable through Helm values and the `COSMOS_EMBED_OSS_IMAGE_REPOSITORY` / `COSMOS_EMBED_OSS_IMAGE_TAG` variables.
