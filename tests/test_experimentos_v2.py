"""Protocolo v2: partición por grupos congelada, pesos fruta-calidad y bootstrap por grupos."""
from pathlib import Path
import sys
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.base import cargar_base
from src.models.experimentos_base_ampliada import (
    CALIDADES, bootstrap_grupos, metricas, particion_desarrollo_prueba, pesos_fruta_calidad)


class TestExperimentosV2(unittest.TestCase):
    def test_particion_congelada_reproducible_y_sin_grupos_compartidos(self):
        base, _, estado = cargar_base(RAIZ)
        etapa, _, elegido = particion_desarrollo_prueba(base)
        guardada = pd.read_csv(RAIZ / 'data/base/particion_desarrollo_prueba.csv')
        self.assertEqual(guardada.sha_base.iloc[0], estado['sha_base'])
        self.assertEqual(guardada.etapa.tolist(), list(etapa))
        dev = set(base.grupo_particion[etapa == 'desarrollo'])
        test = set(base.grupo_particion[etapa == 'prueba'])
        self.assertFalse(dev & test)
        prueba = base[etapa == 'prueba']
        self.assertEqual(set(prueba.calidad_revisada), set(CALIDADES))
        # Ninguna campaña con más del 10 % de la base entra en la prueba.
        fraccion = base.grupo_particion.value_counts() / len(base)
        self.assertTrue((fraccion[list(test)] <= 0.10).all())

    def test_pesos_fruta_calidad_igualan_masa_por_celda(self):
        fruta = np.array(['banano'] * 8 + ['banano'] * 2 + ['limon'] * 5)
        y = np.array(['intermedia'] * 8 + ['buena'] * 2 + ['buena'] * 5)
        w = pesos_fruta_calidad(fruta, y)
        self.assertAlmostEqual(w.sum(), len(y))
        masas = pd.Series(w).groupby(pd.Series(fruta) + '|' + pd.Series(y)).sum()
        self.assertTrue(np.allclose(masas, masas.iloc[0]))

    def test_bootstrap_por_grupos_coincide_con_metricas_completas(self):
        rng = np.random.default_rng(0)
        real = rng.choice(CALIDADES, 60)
        pred = np.where(rng.random(60) < .6, real, rng.choice(CALIDADES, 60))
        grupos = np.repeat(np.arange(12), 5)
        ic = bootstrap_grupos(real, pred, grupos, n=300)
        f1 = metricas(real, pred)['f1_macro']
        self.assertAlmostEqual(f1, f1_score(real, pred, labels=CALIDADES, average='macro', zero_division=0))
        self.assertLessEqual(ic['f1_macro_ic95'][0], ic['f1_macro_ic95'][1])
        self.assertTrue(0 <= ic['f1_macro_ic95'][0] <= 1)


if __name__ == '__main__':
    unittest.main()
