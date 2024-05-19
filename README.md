## Table of Variants and Architectures Assessed

| Variant  | Architecture | Description                                       |
|----------|--------------|---------------------------------------------------|
| Default  | ResNet 18    | Standard ResNet-18 architecture.                  |
| Default  | ResNet 50    | Standard ResNet-50 architecture.                  |
| LIF      | ResNet 18    | ResNet-18 architecture with Leaky Integrate-and-Fire (LIF) neurons. |
| LIF      | ResNet 50    | ResNet-50 architecture with Leaky Integrate-and-Fire (LIF) neurons. |
| ParaLIF  | ResNet 18    | ResNet-18 architecture with Parallel Leaky Integrate-and-Fire (ParaLIF) neurons. |
| ParaLIF  | ResNet 50    | ResNet-50 architecture with Parallel Leaky Integrate-and-Fire (ParaLIF) neurons. |

## Descriptions
ResNet-18 for quick indication of progress really, **ResNet-50** is the true benchmark we should be looking at.

- **Default**: The standard architecture of ResNet without any modifications.
- **LIF**: An architecture variant where decoder neurons are replaced with Leaky Integrate-and-Fire (LIF) neurons.
- **ParaLIF**: A variant similar to LIF, where decoder neurons are Parallel Leaky Integrate-and-Fire (ParaLIF) neurons.

## Performance on CIFAR-10

| Variant  \ Architecture | ResNet 18 | ResNet 50 |
|--------------------------|-----------|-----------|
| Default                  |          %92.1 (@5) |          %94.1 (@5) |
| LIF                      |          %92.2 (@5, @5)|           |
| ParaLIF                  |          %61.0 (@5, @20)|           |
