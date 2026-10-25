#!/usr/bin/env bash
# Reproduce inference on test360_ov3. Run from the upstream PSELDNets repo root.
set -euo pipefail

python src/infer.py \
  experiment=synth_maccdoa \
  data=synth_eval_only \
  mode=test \
  ckpt_path=ckpts/mACCDOA-HTSAT-0.567.ckpt \
  model.kwargs.pretrained_path=null \
  model.batch_size=8 \
  num_workers=4 \
  compile=false \
  sed_threshold="${SED_THRESHOLD:-0.5}"
