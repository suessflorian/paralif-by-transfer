# Deployment

We're playing around with some V100 GPU's in GCP. This comes specifically for training larger models such as the ViT variants on the same Cifar-100 dataset. But interesting would also be the training over the ILSVRC subset of ImageNet.

## Approach

Using [ADC](https://cloud.google.com/docs/authentication/provide-credentials-adc), we can simply spin up in our desired project a V100 instance (provided you have enquired GCP for an increase `GLOBAL_GPU_QUOTA`.

```sh
terraform apply
```

At this point, we can copy our scripts up via `scp`, it is recommend to defer all `ssh` frivelties to `gcloud ssh`/`gcloud scp`. Although we do boot the instance up using a deep learning specific toolkit bundled image, we can go on to install dependancies using either what this project is using `pipenv` or something like `conda`.

## Notes

I thought originally that we'd need to dockerize our scripts, but I didn't see the light going down this way.
