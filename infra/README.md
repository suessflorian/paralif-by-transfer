# Deployment

We're playing around with some GPU compute resource in GCP. This comes specifically for training larger models such as the ViT variants on the same Cifar-100 dataset...

## Approach

Using [ADC](https://cloud.google.com/docs/authentication/provide-credentials-adc), we can simply spin up into our desired project whatever we need (GPU requires a GCP enquiry for an increase on your project wide `GLOBAL_GPU_QUOTA`).

```sh
terraform apply
```

At this point, we can copy our scripts up via `scp`, it is recommend to defer all `ssh` frivelties to `gcloud ssh`/`gcloud scp`. Although we do boot the instance up using a deep learning specific toolkit bundled image, we can go on to install dependancies using either what this project is using `pipenv`. See `bootstrap.sh` script for an idea.

## Notes
- `V100` seems to be best readily available spot instance. Easy 10x improvement to my M1 Max 10 c-core, 32 g-core @ 64GB Ram (on "mps").
    - Mixed precision training is a must to get that (otherwise only 2-3x improvement).
- The recent data augmentation bumps to push ViT beyond 84% are CPU intensive, maybe more cpu's than `n1-standard-2` (bump `num_workers` to core count).
    - As an alternative we could look at https://github.com/NVIDIA/DALI.
