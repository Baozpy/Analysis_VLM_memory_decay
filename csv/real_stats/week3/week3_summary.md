# Week3 Ablations — Summary
- Significance threshold: |Z| ≥ 1.96

## half_life
- Best: **rand_order_s0**  (Z̄ = 467.90, significant)
- Top-5 by Z̄:
  - rand_order_s0: Z̄=467.90 (|Z|̄=467.90, |Z|max=1851.12)
  - rand_order_s2: Z̄=142.10 (|Z|̄=142.10, |Z|max=1043.09)
  - rand_order_s1: Z̄=129.24 (|Z|̄=129.24, |Z|max=496.50)
  - interleave: Z̄=10.10 (|Z|̄=10.10, |Z|max=23.26)
  - baseline_samples: Z̄=8.60 (|Z|̄=8.60, |Z|max=20.00)

## mixed_effects
- Best: **rand_order_s1**  (Z̄ = 0.14, ns)
- Top-5 by Z̄:
  - rand_order_s1: Z̄=0.14 (|Z|̄=0.32, |Z|max=0.98)
  - rand_order_s0: Z̄=0.11 (|Z|̄=0.30, |Z|max=0.97)
  - shuffle_images_s1: Z̄=0.01 (|Z|̄=0.22, |Z|max=1.01)
  - shuffle_images_s2: Z̄=0.01 (|Z|̄=0.22, |Z|max=1.01)
  - baseline_samples: Z̄=0.01 (|Z|̄=0.22, |Z|max=1.01)

## propagation
- Best: **rand_order_s2**  (Z̄ = 208.70, significant)
- Top-5 by Z̄:
  - rand_order_s2: Z̄=208.70 (|Z|̄=208.70, |Z|max=882.00)
  - rand_order_s0: Z̄=206.28 (|Z|̄=206.28, |Z|max=901.00)
  - rand_order_s1: Z̄=198.37 (|Z|̄=198.37, |Z|max=843.00)
  - blocked: Z̄=188.19 (|Z|̄=188.19, |Z|max=803.00)
  - interleave: Z̄=184.78 (|Z|̄=184.78, |Z|max=779.00)

## Overall pick
- **rand_order_s0** (votes = 1)
