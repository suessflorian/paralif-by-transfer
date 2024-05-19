# Spiking ResNet Architectures/Variants Re: Advaserial Robustness

## Descriptions
ResNet-18 is more for quick indication of progress really, **ResNet-50** is a more interesting benchmark.

- **Default**: The standard architecture of ResNet without any modifications.
- **LIF**: An architecture variant where decoder neurons are replaced with Leaky Integrate-and-Fire (LIF) neurons.
- **ParaLIF**: A variant similar to LIF, where decoder neurons are Parallel Leaky Integrate-and-Fire (ParaLIF) neurons. Implementation of the neuron is pulled [from here](https://github.com/NECOTIS/Parallelizable-Leaky-Integrate-and-Fire-Neuron) and lives copied in `paralif.py`.

### Methodology
- We transfer model weights learnt from being heavily trained on ImageNet via adopting [pre-trained models](https://pytorch.org/vision/stable/models.html).
- We identify the 4 residual layers of ResNet as candidate "encoder depths". Initial depth is known to contain "low-level" features and as the depth progresses, features become more "specialisd"/"high-level" for a task.
- We **freeze** the model up to a depth, by prohibiting a layed computation graph overtop of any frozen layers.
- We then fine-tune this model onto various datasets like `CIFAR-100`, `CIFAR-10`, `FashionMNIST` and so on via native backpropogation.
- At some point, we then branch off of this model to either a *LIF* or *ParaLIF* variant, where we replace the model decoder part into either a LIF or ParaLIF part.
- We train via normal backpropogation through time with fixed number of steps of the observation window at 20.

## Performance
_The (@X, @Y) represents how long the training process took; on continuous model for X epochs, and Y epochs on the branched LIF/ParaLIF model_.

- We do have checkpoint files for everything, so these are all readily verifiable.
- _Coming soon_, for branched ResNet's (LIF/ParaLIF) variants, we're also interested in the standard deviation of all results.
    - This is will be found via setting manual seeds (see [`torch.manual_seed`](https://pytorch.org/docs/stable/generated/torch.manual_seed.html))

### CIFAR-10

| Variant  \ Architecture | ResNet 18 | ResNet 50 |
|--------------------------|-----------|-----------|
| Default                  |          %92.1 (@5) |          %94.1 (@5) |
| LIF                      |          %92.2 (@5, @5)|       %94.4 (@5, @5)|
| ParaLIF                  |          %61.0 (@5, @20)|          (@5, @5) |

### CIFAR-100
| Variant  \ Architecture | ResNet 18 | ResNet 50 |
|--------------------------|-----------|-----------|
| Default                  |           |           |
| LIF                      |          |       |
| ParaLIF                  |          |          |

### Fashion-MNIST
| Variant  \ Architecture | ResNet 18 | ResNet 50 |
|--------------------------|-----------|-----------|
| Default                  |           |          |
| LIF                      |          |       |
| ParaLIF                  |          |       |

## Robustness Measures
As this is where our focus predominately lies, we provide a pipeline of well known attacks at various intensities and contrast measure robustness.
