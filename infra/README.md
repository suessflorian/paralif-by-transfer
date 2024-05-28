# Deployment

We're playing around with some GPU compute resource in GCP. This comes specifically for training larger models such as the ViT variants on the same Cifar-100 dataset, but hopefully paves a way for ImageNet. We've only just started looking "what is possible" from the 26th May...

## Benchmarking
Training ViT B16's (weightless), re-structured for CIFAR-100. Various "batch sizes", optimal found automatically with step of 8.

`M1 Max (10 core CPU, 32 core GPU)` (64GB unified mem).
```
Documents/paralif-by-transfer % pipenv run python main.py benchmark --batch 128
Files already downloaded and verified
Files already downloaded and verified
Benchmark: 100%|████████████████████████████| 50048/50048 [26:02<00:00, 32.03images/s, train_accuracy=0.11]
Evaluation: 100%|██████████████████████████████████| 79/79 [01:27<00:00,  1.11s/batch, test_accuracy=0.161]
```

NVIDIA `T4` equipped `n1-standard-2` (2vCPU, 2 core, 8GB mem)
```
(base) floriansuess@florians-deeplearning:~$ python main.py benchmark --batch 64 --device cuda
Files already downloaded and verified
Files already downloaded and verified
Benchmark: 100%|████████████████████████| 50048/50048 [18:52<00:00, 44.19images/s, train_accuracy=0.13]
Evaluation: 100%|████████████████████████████| 157/157 [00:46<00:00,  3.38batch/s, test_accuracy=0.182]
```

## Approach

Using [ADC](https://cloud.google.com/docs/authentication/provide-credentials-adc), we can simply spin up into our desired project whatever we need (GPU requires a GCP enquiry for an increase on your project wide `GLOBAL_GPU_QUOTA`).

```sh
terraform apply
```

At this point, we can copy our scripts up via `scp`, it is recommend to defer all `ssh` frivelties to `gcloud ssh`/`gcloud scp`. Although we do boot the instance up using a deep learning specific toolkit bundled image, we can go on to install dependancies using either what this project is using `pipenv` or something like `conda`.

## Notes

I thought originally that we'd need to dockerize our scripts, but I didn't see the light going down this way.
