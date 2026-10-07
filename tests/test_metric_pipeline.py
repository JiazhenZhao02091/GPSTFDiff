import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
import torch.nn.functional as F

from src.metrics import RMSE, ERGAS, PSNRONE
from src.metrics._utils import get_valid_mask
from tools._metric_io import (
    global_metric_value, load_image_tensor, load_mask_tensor,
    match_image_paths, mean_defined,
)


ROOT = Path(__file__).resolve().parents[1]


class QuietLogger:
    def __init__(self, **kwargs):
        pass

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def load_definitions(path, class_methods=None):
    tree = ast.parse((ROOT / path).read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    if class_methods is not None:
        for node in definitions:
            if isinstance(node, ast.ClassDef):
                node.body = [method for method in node.body
                             if isinstance(method, ast.FunctionDef) and method.name in class_methods]
    namespace = dict(Path=Path, torch=torch, F=F, np=np, pd=pd, time=time,
                     FusionLogger=QuietLogger, get_valid_mask=get_valid_mask,
                     global_metric_value=global_metric_value, load_image_tensor=load_image_tensor,
                     load_mask_tensor=load_mask_tensor, match_image_paths=match_image_paths,
                     mean_defined=mean_defined)
    import os
    namespace['os'] = os
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


Tracker = load_definitions('src/logger/tracker.py')['Tracker']
offline_report = load_definitions('tools/offline_metric_cal.py')['offline_evaluation_report']
batch_report = load_definitions('tools/offline_metric_cal_by_dict.py')['batch_offline_evaluation_and_report']


class MetricPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gt_dir = self.root / 'Landsat_02'
        self.pred_dir = self.root / 'pred'
        self.gt_dir.mkdir()
        self.pred_dir.mkdir()

    def write_pair(self, gt, pred, key='20200101', mask=None, pred_layout='HWC'):
        tiff.imwrite(self.gt_dir / f'{key}_L_0.tif', gt, photometric='minisblack')
        if pred_layout == 'CHW':
            pred = pred.transpose(2, 0, 1)
        tiff.imwrite(self.pred_dir / f'{key}_save_img_0.tif', pred, photometric='minisblack')
        if mask is not None:
            mask_dir = self.root / 'mask'
            mask_dir.mkdir(exist_ok=True)
            tiff.imwrite(mask_dir / f'{key}_mask_0.tif', mask)

    def report(self, metric_list, **kwargs):
        return offline_report(self.gt_dir, self.pred_dir, 'test', metric_list,
                              self.root / 'output', **kwargs)

    def test_cia_lgc_without_mask(self):
        g = np.full((16, 16, 2), 2000, dtype=np.uint16)
        p = np.full_like(g, 4000)
        self.write_pair(g, p)
        result = self.report([RMSE(), ERGAS()])
        self.assertAlmostEqual(result['RMSE'], 0.2, places=6)
        self.assertAlmostEqual(result['ergas'], 6.25, places=5)

    def test_ml_float_mask_discovery(self):
        g = np.full((16, 16, 2), 0.2, dtype=np.float32)
        p = np.full_like(g, 0.4)
        mask = np.ones((16, 16), dtype=np.uint8)
        mask[:8] = 0
        g[:8] = np.nan
        p[:8] = np.inf
        self.write_pair(g, p, mask=mask)
        result = self.report([RMSE(), ERGAS()], normalize_scale=1)
        self.assertAlmostEqual(result['RMSE'], 0.2, places=6)
        self.assertAlmostEqual(result['ergas'], 6.25, places=5)

    def test_chw_prediction_and_explicit_mask(self):
        g = np.full((16, 16, 2), 0.2, dtype=np.float32)
        p = g + 0.1
        self.write_pair(g, p, mask=np.ones((16, 16), dtype=np.uint8), pred_layout='CHW')
        result = self.report([RMSE()], normalize_scale=1, pred_layout='CHW',
                             mask_dir_path=self.root / 'mask')
        self.assertAlmostEqual(result['RMSE'], 0.1, places=6)

    def test_pairing_rejects_equal_count_wrong_ids(self):
        g = np.zeros((16, 16, 2), dtype=np.float32)
        self.write_pair(g, g)
        pred = self.pred_dir / '20200101_save_img_0.tif'
        pred.rename(self.pred_dir / '20200201_save_img_0.tif')
        with self.assertRaisesRegex(ValueError, 'identifiers differ'):
            self.report([RMSE()])

    def test_global_rmse_keeps_band_results(self):
        g = np.zeros((16, 16, 2), dtype=np.float32)
        p = np.ones_like(g)
        p[:, :, 1] = 3
        self.write_pair(g, p)
        result = self.report([RMSE(is_reduce_channel=False)], normalize_scale=1)['RMSE']
        self.assertAlmostEqual(result['global'], np.sqrt(5), places=6)
        self.assertEqual(result['bands'], [1, 3])

    def test_global_psnr_uses_pooled_mse(self):
        g = np.zeros((16, 16, 2), dtype=np.float32)
        p = np.ones_like(g)
        p[:, :, 1] = 3
        self.write_pair(g, p)
        metric = PSNRONE(max_value=1, is_reduce_channel=False)
        result = self.report([metric], normalize_scale=1)['PSNRONE']
        self.assertAlmostEqual(result['global'], -10 * np.log10(5), places=5)
        self.assertFalse(metric.is_reduce_channel)

    def test_empty_mask_does_not_poison_other_images(self):
        g = np.full((16, 16, 2), 0.2, dtype=np.float32)
        self.write_pair(g, g + 10, key='20200101', mask=np.zeros((16, 16), dtype=np.uint8))
        self.write_pair(g, g + 0.1, key='20200201', mask=np.ones((16, 16), dtype=np.uint8))
        result = self.report([RMSE()], normalize_scale=1)
        self.assertAlmostEqual(result['RMSE'], 0.1, places=6)

    def test_batch_report_uses_masks_and_standard_global_rmse(self):
        g = np.zeros((16, 16, 2), dtype=np.float32)
        p = np.ones_like(g)
        p[:, :, 1] = 3
        mask = np.ones((16, 16), dtype=np.uint8)
        mask[:8] = 0
        p[:8] = 100
        self.write_pair(g, p, mask=mask)
        output = self.root / 'batch'
        batch_report({'test': self.pred_dir}, self.gt_dir, output, 'metrics.txt',
                     [RMSE(is_reduce_channel=False)], normalize_scale=1)
        self.assertIn('RMSE: 2.2361', (output / 'metrics.txt').read_text())
        self.assertTrue((output / 'metrics.tex').exists())

    def test_tracker_skips_undefined_but_preserves_perfect_psnr(self):
        tracker = Tracker('RMSE', 'PSNR')
        tracker.update('RMSE', float('nan'))
        self.assertTrue(np.isnan(tracker.results['RMSE']))
        tracker.update('RMSE', 0.1)
        tracker.update('RMSE', float('nan'))
        self.assertAlmostEqual(tracker.results['RMSE'], 0.1)
        tracker.update('PSNR', float('inf'))
        self.assertTrue(np.isinf(tracker.results['PSNR']))

    def test_gpstfdiff_inference_routes_optional_mask(self):
        cls = load_definitions('src/inferencer/GPSTFDiff_inferencer.py',
            {'inference', 'before_inference_iter', 'get_model_sampling_input', 'inference_iter'})['Inferencer']
        g = torch.full((1, 2, 16, 16), 0.2)
        p = torch.full_like(g, 0.4)
        p[:, :, :8] = 0.9
        mask = torch.ones(1, 1, 16, 16)
        mask[:, :, :8] = 0
        for use_mask in (False, True):
            with self.subTest(use_mask=use_mask):
                exp = cls()
                exp.device = 'cpu'
                exp.model = SimpleNamespace(eval=lambda: None, sample=lambda *args: p * 2 - 1)
                exp.metric_list = [RMSE(), ERGAS()]
                exp.inference_tracker = Tracker('loss', 'RMSE', 'ergas')
                exp.txt_logger = QuietLogger()
                exp.backend_logger = SimpleNamespace(add_scalar=lambda *args: None)
                exp.inference_imgs_dir = self.root / 'images'
                exp.img_save = exp.img_show = lambda *args: None
                exp.current_epoch = exp.current_inference_step = 0
                data = {name: g * 2 - 1 for name in
                        ('fine_img_01', 'fine_img_02', 'coarse_img_01', 'coarse_img_02')}
                data.update(key=['20200101_L_0'], dataset_name=['ML' if use_mask else 'CIA'],
                            normalize_scale=torch.tensor([1.0]), normalize_mode=torch.tensor([2]))
                if use_mask:
                    data['mask'] = mask
                exp.test_dataloader = [data]
                exp.inference()
                expected_mask = mask if use_mask else None
                self.assertAlmostEqual(exp.inference_tracker.results['RMSE'], RMSE()(g, p, expected_mask).item(), places=6)
                self.assertAlmostEqual(exp.inference_tracker.results['ergas'], ERGAS()(g, p, expected_mask).item(), places=5)

    def test_starfm_mask_patch_reconstruction_excludes_padding(self):
        cls = load_definitions('src/inferencer/starfm_inferencer.py',
            {'get_metric_mask_patches', 'cal_img_padding_hw', 'cal_img_padding_pixel_num_hw',
             'cal_patch_num', 'cal_img_padding', 'after_inference_iter'})['Inferencer']
        exp = cls()
        exp.patch_size = exp.patch_stride = 12
        exp.window_size = 3
        exp.virtual_patch_size = 14
        g = torch.ones(1, 2, 22, 22)
        mask = torch.ones(1, 22, 22)
        mask[:, 8:10, 8:10] = 0
        patches = exp.get_metric_mask_patches({'fine_img_02': g, 'mask': mask})
        h, w = exp.cal_img_padding_hw(22, 22)
        full, _ = exp.after_inference_iter(patches.reshape(1, -1, patches.shape[-1]), patches, h, w, None)
        self.assertEqual(full.shape[-2:], (24, 24))
        torch.testing.assert_close(full[:, :, 1:23, 1:23], mask[:, None].expand_as(g))
        self.assertEqual(full.sum().item(), mask.sum().item() * 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
