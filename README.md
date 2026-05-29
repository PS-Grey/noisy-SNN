# Making Convolutional SNNs Work with a Single Line of Code

Inspired by the quantum stochasticity of the brain I decided to test an idea. Which led me to an unexpected finding.

Adding membrane noise during SNN training isn't just helpful. For convolutional SNNs it's the difference between a working network and random chance.

```python
membrane = membrane + torch.randn_like(membrane) * noise_std
```

| Architecture | Fashion-MNIST | CIFAR-10 |
|-------------|--------------|----------|
| Conv-SNN (no noise) | 16.5% | 11.7% |
| Conv-SNN (with noise) | 88.6% | 73.1% |
| CNN-ANN baseline | 93.9% | 88.6% |

16.5% and 11.7% on a 10-class task is random chance. The network learned nothing. One line of code fixes it.

And it doesn't just fix accuracy. It improves everything:

| Metric | Control SNN | Noisy SNN | Change |
|--------|------------|-----------|--------|
| Accuracy (MNIST) | 92.1% | 97.9% | +5.8% |
| Activation rate | 0.43 | 0.30 | -30% (sparser) |
| Error correlation | 0.86 | 0.74 | -14% (more diverse) |

More accurate, more energy efficient, more diverse. Three birds one stone.

## What noise is best?

Tested 4 strategies on CIFAR-10:

| Strategy | Best Accuracy | How it works |
|----------|-------------|-------------|
| **Signal-relative (α=0.6)** | **73.1%** | Noise scales with signal strength at each layer |
| Threshold-relative (α=0.6) | 69.0% | Noise scales with firing threshold |
| Fixed (0.6) | 67.5% | Same noise everywhere |
| Learned (init=0.4) | 61.1% | Network decides its own noise level |
| Control (no noise) | 11.7% | Nothing works |

Signal-relative wins. `noise_std = α × √(batch_norm_running_variance)`. One hyperparameter.

## Experiments

| File | What it tests |
|------|--------------|
| `snn_ensemble_emergence.py` | Flat SNN ensembles, noise vs control, soft voting emergence |
| `snn_diverse_triage.py` | N=2 diverse SNN + ANN escalation triage |
| `snn_conv_experiment.py` | Conv-SNN with structural diversity and hard core analysis |
| `snn_arch_sweep.py` | Architectural parameter sweep (timesteps, beta, hidden size, threshold) |
| `snn_noise_strategies_cifar10.py` | Four noise scaling strategies on CIFAR-10 |
| `snn_bn_warmup_test.py` | Disambiguation test for signal-relative vs BN warmup schedule |

All experiments run on PyTorch. Colab T4 GPU is sufficient. Fashion-MNIST and CIFAR-10.

## What's next

- Head-to-head against established SNN training techniques (learnable thresholds, tdBN, residual connections)
- CIFAR-100 as third dataset
- Higher α values (sweep hasn't found the peak yet)
- Neuromorphic hardware validation

## Full writeup

[Medium article](https://medium.com/@sergiyyun/inspired-by-quantum-stochasticity-of-the-brain-i-decided-to-test-an-idea-a9cf6b425b04)



