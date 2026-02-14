# CoAtNet Paper Reproduction(2021)

## Introduction
This project reproduces the CoAtNet architecture as described in the original paper, while tuning the original architecture to suit Tiny Image Net to prevent overfitting.

## Architecture Summary
| Stage | Layer / Block              | Output Shape     | Notes                                 |
| ----: | -------------------------- | ---------------- | ------------------------------------- |
| Input | Input                      | `(224, 224, 3)`  | RGB image                             |
|     1 | Conv2D, 64, 3×3, stride 2  | `(112, 112, 64)` | Patch embedding                       |
|       | BatchNorm + GELU           | `(112, 112, 64)` |                                       |
|     2 | MBConv (64 → 96)           | `(112, 112, 96)` | Local feature extraction              |
|     3 | Conv2D, 96, 3×3, stride 2  | `(56, 56, 96)`   | Downsampling                          |
|     4 | MBConv (96 → 192)          | `(56, 56, 192)`  |                                       |
|     5 | Conv2D, 192, 3×3, stride 2 | `(28, 28, 192)`  | Downsampling                          |
|     6 | Reshape                    | `(784, 192)`     | Flatten spatial tokens                |
|     7 | Self-Attention             | `(784, 192)`     | `window_size=28`, `heads=12` (global) |
|     8 | Reshape                    | `(28, 28, 192)`  | Restore spatial layout                |
|     9 | Conv2D, 384, 3×3, stride 2 | `(14, 14, 384)`  | Downsampling                          |
|    10 | Reshape                    | `(196, 384)`     |                                       |
|    11 | Self-Attention             | `(196, 384)`     | `window_size=14`, `heads=12`          |
|    12 | Reshape                    | `(14, 14, 384)`  |                                       |
|    13 | Conv2D, 384, 3×3, stride 2 | `(7, 7, 384)`    | Final spatial reduction               |
|    14 | Global Average Pooling     | `(384)`          |                                       |
|    15 | Dense (Softmax)            | `(200)`          | TinyImageNet classes                  |

The original CoAtNet-0 model has 2, 3, 5, 2 blocks on S1 (MBConv),S2 (MBConv),S3 (Transformer) and S4 (Transformer) with ~25M parameters, while we assign only 1 block to each stage to reduce overfitting problem, which gives us ~3M parameters.


## Dataset
tiny-imagenet-200 is used as the dataset in this reproduction. The images are augmented aligns with the original paper (random resized crop, random horizontal flip, cutmix, mixup, erase, autoaugment and label smoothing).

## Results
We train CoAtNet for 80 epochs

| Metric            | Value |
| ----------------- | ----- |
| Training accuracy | 95.89% |
| Test accuracy     | 44.80% |

The training loss for 80 epochs for CoAtNet is below

![](image/TrainingLoss.png)


## Discussion
### Overfitting Problem
From the result, we can see that the model still has overfitting problems on Tiny ImageNet Dataset. For comparison, ConvNeXt we trained earlier has 78.81% and 60.17% accuracy. One of the reasons is that transformer blocks can correlate any two patches in a layer, which enables it to learn long-range patterns. Although it has MBConv blocks at early stages, it only recognise low-level patterns and add some inductive bias. Furthermore, MBConv blocks don't limit what happens next. As a result, once it reaches transformer stages, the model starts memorising patterns. Moreover, Tiny ImageNet only has 100k images. Thus, there aren't enough samples to rule out bad hypothesis and spurious patterns are not likely to get contradicted during training. Consequently, our model has overfitting problems.

### Future Work : Moving Toward C-C-C-T
In the future, we can add more MBConv blocks to process features before pushing the model into transformer stage. For example, we can check if C-C-C-T works better, as it keeps doing convolution until the feature map is really small (7 * 7). Until then, the transformer block only has 49 patches to correlate, which is likely to reduce overfitting. Moreover, the 14 * 14 convolution stage can also be added more blocks for example a 1 : 1 : 3 : 1 ratio according to Liu et al. (2022).

## References
Dai, Z., Liu, H., Le, Q. V., & Tan, M. (2021). CoAtNet: Marrying convolution and attention for all data sizes. Advances in Neural Information Processing Systems, 34, 3965–3977. https://arxiv.org/abs/2106.04803

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Tiny ImageNet Dataset
Wu, J., Zhang, J., Xie, Y., & others. (2017).
Tiny ImageNet Visual Recognition Challenge.
Stanford University.
https://tiny-imagenet.herokuapp.com/
