from importlib import import_module


_MODELS = {
    'SwinSTFM': '.swinstf',
    'OPGANGenerator': '.opgan',
    'OPGANDiscriminator': '.opgan',
    'MSDiscriminator': '.ganstfm',
    'SFFusion': '.ganstfm',
    'STFDCNN': '.stfdcnn',
    'STFGANDiscriminator': '.stfgan',
    'STFGANGenerator': '.stfgan',
    'GaussianDiffusion': '.stfdiff',
    'PredNoiseNet': '.stfdiff',
    'BiGaussianDiffusion': '.stfdiff',
    'BiPredNoiseNet': '.stfdiff',
}

__all__ = list(_MODELS)


def __getattr__(name):
    if name not in _MODELS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(_MODELS[name], __name__), name)
    globals()[name] = value
    return value
