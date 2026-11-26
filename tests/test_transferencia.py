"""Aislamiento del extractor y de los aumentos de validación."""
from pathlib import Path
import sys
import unittest
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'), 'CNN opcional no instalada.')
class TestTransferencia(unittest.TestCase):
    def test_extractor_sigue_fijo_en_entrenamiento(self):
        import torch
        from src.models.cnn_calidad import CNNTransferencia, ConfigTransferencia
        anterior = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, anterior)
        torch.set_num_threads(2)
        modelo = CNNTransferencia(ConfigTransferencia()).train()
        self.assertFalse(modelo.extractor.training)
        self.assertFalse(any(p.requires_grad for p in modelo.extractor.parameters()))
        antes = {k: v.clone() for k, v in modelo.extractor.state_dict().items()}
        modelo(torch.randn(2, 3, 224, 224)).sum().backward()
        self.assertTrue(all(p.grad is None for p in modelo.extractor.parameters()))
        self.assertTrue(any(p.grad is not None for p in modelo.cabezal.parameters()))
        self.assertTrue(all(torch.equal(v, antes[k]) for k, v in modelo.extractor.state_dict().items()))

    def test_evaluacion_no_usa_vistas_aumentadas(self):
        import numpy as np
        import torch
        from src.models.cnn_calidad import ConfigTransferencia, entrenar_cabezal
        anterior = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, anterior)
        torch.set_num_threads(2)
        torch.manual_seed(11)
        z = torch.randn(12, 5, 576)
        y = np.array([0, 1, 2] * 4)
        cfg = ConfigTransferencia(max_epocas=4, min_epocas=1, paciencia=2)
        otro = z.clone()
        otro[:, 1:] = 10000
        a, _, ia = entrenar_cabezal(z, y, cfg, 42, z, y)
        b, _, ib = entrenar_cabezal(z, y, cfg, 42, otro, y)
        self.assertEqual(ia, ib)
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
