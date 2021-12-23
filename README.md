# pytorch-lightning-test-case

Use the canonical [MNIST](http://yann.lecun.com/exdb/mnist/) dataset to construct a GAN in Lightning 
that is capable of reproducing handwritten digits.

----
PyTorch Lightning vs. Vanilla
1. Focus on high-level experimentation
- performance and bottleneck profiling
- logging
- metrics
- visualization
2. Utilize features for efficient training
- model checkpointing
- early stopping
- gradient clipping
3. Don't worry about hardware
- running of code on any hardware
- easy distributed training
- 16-bit precision