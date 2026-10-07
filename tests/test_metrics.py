import unittest

import numpy as np
import torch
from skimage.metrics import structural_similarity

from src.metrics import RMSE, MAE, PSNR, PSNRONE, SSIM, ERGAS, CC, SAM, UIQI, RMSE_BD


def metrics():
    return [RMSE(), MAE(), PSNR(max_value=1), PSNRONE(max_value=1),
            SSIM(data_range=1), ERGAS(), CC(), SAM(), UIQI()]


class MetricTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.gt = torch.rand(2, 3, 24, 24) * 0.7 + 0.1
        self.pred = self.gt + torch.randn_like(self.gt) * 0.03

    def test_standard_formulas(self):
        g = self.gt.numpy().astype(np.float64)
        p = self.pred.numpy().astype(np.float64)
        mse = np.mean((g - p) ** 2)
        expected_ergas = np.mean(100 / 16 * np.sqrt(np.mean(
            np.mean((g - p) ** 2, axis=(2, 3)) / np.mean(g, axis=(2, 3)) ** 2, axis=1)))
        expected_cc = np.mean([np.corrcoef(g[b, c].ravel(), p[b, c].ravel())[0, 1]
                               for b in range(2) for c in range(3)])
        expected_sam = np.arccos(np.clip(np.sum(g * p, axis=1)
            / (np.linalg.norm(g, axis=1) * np.linalg.norm(p, axis=1)), -1, 1)).mean()
        expected_ssim = np.mean([structural_similarity(g[b, c], p[b, c], data_range=1,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            for b in range(2) for c in range(3)])
        vg, vp = g.var(axis=(2, 3)), p.var(axis=(2, 3))
        mg, mp = g.mean(axis=(2, 3)), p.mean(axis=(2, 3))
        cov = ((g - mg[..., None, None]) * (p - mp[..., None, None])).mean(axis=(2, 3))
        expected_uiqi = (4 * cov * mg * mp / ((vg + vp) * (mg ** 2 + mp ** 2))).mean()
        cases = [(RMSE(), np.sqrt(mse)), (MAE(), np.abs(g - p).mean()),
                 (PSNR(max_value=1), -10 * np.log10(mse)), (ERGAS(), expected_ergas),
                 (CC(), expected_cc), (SAM(), expected_sam), (SSIM(data_range=1), expected_ssim),
                 (UIQI(), expected_uiqi)]
        for metric, expected in cases:
            with self.subTest(metric=metric.__name__):
                self.assertAlmostEqual(metric(self.gt, self.pred).item(), expected, places=5)

    def test_full_mask_matches_standard(self):
        mask = torch.ones(2, 1, 24, 24)
        for metric in metrics():
            with self.subTest(metric=metric.__name__):
                torch.testing.assert_close(metric(self.gt, self.pred, mask),
                                           metric(self.gt, self.pred), atol=1e-6, rtol=1e-5)

    def test_invalid_nan_and_inf_do_not_change_metrics(self):
        mask = torch.ones(2, 1, 24, 24)
        mask[:, :, :5, :5] = 0
        g, p = self.gt.clone(), self.pred.clone()
        g[:, :, :5, :5] = float('nan')
        p[:, :, :5, :5] = float('inf')
        for metric in metrics():
            with self.subTest(metric=metric.__name__):
                torch.testing.assert_close(metric(g, p, mask), metric(self.gt, self.pred, mask))

    def test_mask_layouts_and_per_band_support(self):
        mask = torch.ones(2, 24, 24)
        mask[:, :8] = 0
        for metric in metrics():
            with self.subTest(metric=metric.__name__):
                torch.testing.assert_close(metric(self.gt, self.pred, mask),
                                           metric(self.gt, self.pred, mask[:, None]))
        mask = mask[:, None].expand_as(self.gt).clone()
        mask[:, 1] = 0
        for cls, kwargs in [(RMSE, {}), (MAE, {}), (PSNRONE, {'max_value': 1}),
                            (SSIM, {'data_range': 1}), (CC, {}), (UIQI, {})]:
            result = cls(is_reduce_channel=False, **kwargs)(self.gt, self.pred, mask)
            self.assertEqual(tuple(result.shape), (3,))
            self.assertTrue(torch.isnan(result[1]))
            self.assertTrue(torch.isfinite(result[[0, 2]]).all())

    def test_empty_mask_is_undefined(self):
        mask = torch.zeros(2, 1, 24, 24)
        for metric in metrics() + [RMSE_BD()]:
            with self.subTest(metric=metric.__name__):
                self.assertTrue(torch.isnan(metric(self.gt, self.pred, mask)))
        for cls, kwargs in [(RMSE, {}), (MAE, {}), (PSNRONE, {'max_value': 1})]:
            value = cls(is_reduce_channel=False, **kwargs)(self.gt, self.pred, mask)
            self.assertEqual(tuple(value.shape), (3,))
            self.assertTrue(torch.isnan(value).all())

    def test_masked_error_uses_valid_count(self):
        g = torch.zeros(1, 2, 12, 12)
        p = torch.full_like(g, 99)
        mask = torch.zeros_like(g)
        mask[:, 0, 0, :2] = 1
        mask[:, 1, 0, 0] = 1
        p[:, 0, 0, :2] = 1
        p[:, 1, 0, 0] = 3
        self.assertAlmostEqual(RMSE()(g, p, mask).item(), np.sqrt(11 / 3), places=6)
        self.assertAlmostEqual(MAE()(g, p, mask).item(), 5 / 3, places=6)

    def test_perfect_psnr_and_missing_band(self):
        mask = torch.ones_like(self.gt)
        mask[:, 1] = 0
        result = PSNRONE(max_value=1, is_reduce_channel=False)(self.gt, self.gt, mask)
        self.assertTrue(torch.isinf(result[[0, 2]]).all())
        self.assertTrue(torch.isnan(result[1]))
        self.assertTrue(torch.isinf(PSNR(max_value=1)(self.gt, self.gt)))

    def test_sam_scale_and_zero_norm(self):
        g = torch.tensor([0.2, 0.3, 0.7]).view(1, 3, 1, 1)
        p = torch.tensor([0.4, 0.1, 0.6]).view(1, 3, 1, 1)
        expected = SAM()(g, p)
        torch.testing.assert_close(SAM()(g * 1e-5, p * 1e-5), expected)
        self.assertLess(SAM()(g * 1e-5, g * 1e-5).item(), 1e-7)
        self.assertTrue(torch.isnan(SAM()(torch.zeros_like(g), p)))
        torch.testing.assert_close(SAM(unit='degree')(g, p), torch.rad2deg(expected))
        g = torch.cat((g, torch.zeros_like(g)), dim=-1)
        p = torch.cat((p, torch.zeros_like(p)), dim=-1)
        torch.testing.assert_close(SAM()(g, p), expected)

    def test_sam_requires_all_bands_valid(self):
        g = torch.tensor([[[[1., 1.]], [[0., 0.]]]])
        p = torch.tensor([[[[1., 0.]], [[0., 1.]]]])
        mask = torch.ones_like(g)
        mask[:, 1, :, 1] = 0
        self.assertEqual(SAM()(g, p, mask).item(), 0)

    def test_cc_low_variance(self):
        for amplitude in (1e-3, 1e-4, 1e-5):
            x = 0.5 + torch.linspace(-amplitude, amplitude, 256).view(1, 1, 16, 16)
            self.assertAlmostEqual(CC()(x, 2 * x - 0.5).item(), 1, places=6)
        self.assertTrue(torch.isnan(CC()(torch.ones_like(x), torch.ones_like(x))))

    def test_uiqi_constant_limits(self):
        g = torch.ones(1, 1, 12, 12)
        self.assertEqual(UIQI()(g, g).item(), 1)
        self.assertEqual(UIQI()(g * 0, g * 0).item(), 1)
        self.assertAlmostEqual(UIQI()(g, g * 2).item(), 0.8, places=6)

    def test_ssim_identical_constant(self):
        mask = torch.ones(1, 1, 24, 24)
        mask[:, :, :5] = 0
        for value in (0.0, 0.01, 0.2, 0.5, 1.0):
            g = torch.full((1, 3, 24, 24), value)
            for valid in (None, mask):
                self.assertAlmostEqual(SSIM(data_range=1)(g, g, valid).item(), 1, places=6)

    def test_ergas_order_and_scale(self):
        g = torch.full((1, 2, 12, 12), 0.2)
        p = g * 2
        self.assertAlmostEqual(ERGAS()(g, p).item(), 6.25, places=5)
        self.assertAlmostEqual(ERGAS()(p, g).item(), 3.125, places=5)
        torch.testing.assert_close(ERGAS()(g * 10000, p * 10000), ERGAS()(g, p))
        self.assertEqual(ERGAS()(g * 0, p * 0).item(), 0)
        self.assertTrue(torch.isinf(ERGAS()(g * 0, p)))

    def test_boundary_rmse(self):
        g = torch.zeros(1, 1, 16, 16)
        p = torch.ones_like(g)
        mask = torch.ones_like(g)
        mask[:, :, 8, 8] = 0
        p[:, :, 8, 8] = float('nan')
        self.assertEqual(RMSE_BD(radius=1)(g, p, mask).item(), 1)
        self.assertTrue(torch.isnan(RMSE_BD()(g, p, torch.ones_like(mask))))
        torch.testing.assert_close(RMSE_BD()(self.gt, self.pred), RMSE()(self.gt, self.pred))

    def test_input_validation_and_dtype(self):
        for metric in metrics():
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(ValueError):
                    metric(self.gt, self.pred[:, :1])
                with self.assertRaises(ValueError):
                    metric(self.gt, self.pred, torch.ones(2, 4, 24, 24))
        self.assertEqual(RMSE()(self.gt.double(), self.pred.double()).dtype, torch.float64)
        self.assertEqual(RMSE()(self.gt.half(), self.pred.half()).dtype, torch.float32)
        with self.assertRaises(ValueError):
            SSIM(gussian_kernel_size=4)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
