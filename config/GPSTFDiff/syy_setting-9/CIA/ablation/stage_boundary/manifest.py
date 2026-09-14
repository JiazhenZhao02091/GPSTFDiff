"""Experiment grid for GPSTFDiff stage-boundary sensitivity."""

SAMPLING_TIMESTEPS = 100
NUM_TRAIN_TIMESTEPS = 1000

# Paper table: compact, interpretable points spanning coarse-heavy to fine-heavy.
PAPER_SWEEP = [
    dict(name="A", t1=50, t2=100),
    dict(name="B", t1=100, t2=150),
    dict(name="C", t1=100, t2=200),
    dict(name="D", t1=100, t2=300),
    dict(name="E", t1=150, t2=300),
    dict(name="F", t1=200, t2=400),
    dict(name="G", t1=300, t2=600),
]

# Heatmap grid: small enough to run, dense enough to show the stable basin.
HEATMAP_K1_VALUES = [5, 10, 15, 20, 30]
HEATMAP_K2_VALUES = [10, 15, 20, 30, 40, 50, 60]

FINAL_SETTING = dict(name="Final", t1=100, t2=300)


def t_from_k(k):
    return int(k * NUM_TRAIN_TIMESTEPS / SAMPLING_TIMESTEPS)


def build_heatmap_sweep():
    experiments = []
    for k1 in HEATMAP_K1_VALUES:
        for k2 in HEATMAP_K2_VALUES:
            if k1 >= k2:
                continue
            experiments.append(
                dict(
                    name=f"K1_{k1:02d}_K2_{k2:02d}",
                    t1=t_from_k(k1),
                    t2=t_from_k(k2),
                )
            )
    return experiments


def all_experiments():
    by_key = {}
    for item in PAPER_SWEEP + build_heatmap_sweep():
        key = (item["t1"], item["t2"])
        by_key.setdefault(key, item)
    return list(by_key.values())

