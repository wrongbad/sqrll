# SQRLL : Simplified Quasi-Recurrent Linear Layer

This is based on ideas from [QRNN](https://github.com/salesforce/pytorch-qrnn) ([Paper](https://arxiv.org/abs/1611.01576))

The "simplification" hereis the removal of the forget-gate multiplying with the input (TODO DIAGRAMS) from the optimized recurrent kernel. One could easily pre-multiply the input to achieve the same behavior has QRNN, but you could also choose any other input gating scheme as well.

Aside from the similar recurrent kernel, I've built an example model that is very different from the original QRNN model. Instead I chose to follow the patterns for modern transformers, with residual skip connections and single-time-step feed-forward diffusion layers.