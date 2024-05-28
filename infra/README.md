# Deployment

We're playing around with some GPU compute resource in GCP. This comes specifically for training larger models such as the ViT variants on the same Cifar-100 dataset...

## Approach

Using [ADC](https://cloud.google.com/docs/authentication/provide-credentials-adc), we can simply spin up into our desired project whatever we need (GPU requires a GCP enquiry for an increase on your project wide `GLOBAL_GPU_QUOTA`).

```sh
terraform apply
```

At this point, we can copy our scripts up via `scp`, it is recommend to defer all `ssh` frivelties to `gcloud ssh`/`gcloud scp`. Although we do boot the instance up using a deep learning specific toolkit bundled image, we can go on to install dependancies using either what this project is using `pipenv`. See `bootstrap.sh` script for an idea.
