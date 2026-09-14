# SA-STF Configs

SA-STF is kept under `compare/SA-STF` as the original comparison repository. These configs provide repository-style entry points that match the other methods:

```bash
python tools/train/train_sastf.py --congfig_path config/sastf/syy_setting-9/CIA/train.py
python tools/inference/test_sastf.py --congfig_path config/sastf/syy_setting-9/CIA/inference.py
python tools/inference/test_sastf.py --congfig_path config/sastf/syy_setting-9/CIA/inference_patch.py
```

Default official weights are read from `compare/SA-STF/weights`. New runs are written under `results/sastf`.
