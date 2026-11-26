"""Pruebas unitarias e integración para validación agrupada, ajuste ML y CNN."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.models.evaluacion import (
    CALIDADES,
    FRUTAS,
    cargar_base,
    consolidar_resultados_oof,
    crear_cv_interno,
    crear_pliegues_externos,
)
from src.models.modelos_tradicionales import evaluar_tradicionales_anidados

TORCH_DISPONIBLE = (
    importlib.util.find_spec('torch') is not None
    and importlib.util.find_spec('torchvision') is not None
)
MATPLOTLIB_DISPONIBLE = importlib.util.find_spec('matplotlib') is not None


def _crear_base_sintetica(raiz: Path, n_grupos: int = 9, fotos_por_grupo: int = 4) -> tuple[pd.DataFrame, list[str], dict]:
    rng = np.random.default_rng(42)
    columnas = [
        *[f'color_f{i}' for i in range(28)],
        *[f'textura_f{i}' for i in range(12)],
        *[f'forma_f{i}' for i in range(3)],
    ]
    assert len(columnas) == 43

    filas = []
    carpeta_cnn = raiz / 'data/processed/sintetico_cnn'
    carpeta_cnn.mkdir(parents=True, exist_ok=True)
    idx = 0
    for g in range(n_grupos):
        calidad = CALIDADES[g % len(CALIDADES)]
        fruta = FRUTAS[g % len(FRUTAS)]
        desplazamiento = CALIDADES.index(calidad) * 1.5
        for _ in range(fotos_por_grupo):
            idx += 1
            ruta_img = carpeta_cnn / f'ej_{idx:03}.png'
            arr = rng.integers(20, 235, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(ruta_img)
            feats = rng.normal(loc=desplazamiento, scale=0.6, size=len(columnas))
            fila = {
                'id_ejemplar': f'ej_{idx:03}',
                'id_origen': f'orig_{idx:03}',
                'codigo_revision': f'rev_{idx:03}',
                'fuente': 'propia' if g % 2 == 0 else 'publica',
                'fruta': fruta,
                'calidad_revisada': calidad,
                'grupo_ejemplar': f'grp_{g:02}',
                'grupo_particion': f'part_{g:02}',
                'sha_original': f'sha_{idx:03}',
                'sha_mascara': f'msha_{idx:03}',
                'sha_cnn': f'csha_{idx:03}',
                'firma_preparacion': 'firma_test',
                'revisor': 'test',
                'ruta_original': str(ruta_img),
                'ruta_mascara': str(ruta_img),
                'ruta_cnn': str(ruta_img),
                'id_fruto_propuesto': '',
                'certeza_identidad': '',
                'revisor_identidad': '',
            }
            for col, val in zip(columnas, feats):
                fila[col] = float(val)
            filas.append(fila)

    df = pd.DataFrame(filas)
    destino = raiz / 'data/base'
    destino.mkdir(parents=True, exist_ok=True)
    ruta_csv = destino / 'datos.csv'
    df.to_csv(ruta_csv, index=False, encoding='utf-8-sig')
    sha = hashlib.sha256(ruta_csv.read_bytes()).hexdigest()
    estado = {'caracteristicas': columnas, 'sha_base': sha}
    (destino / 'estado.json').write_text(json.dumps(estado, indent=2), encoding='utf-8')
    return df, columnas, estado


class TestModelosYValidacion(unittest.TestCase):
    def test_referencia_rechaza_cambios_de_grupo_y_bloque(self) -> None:
        from src.training.entrenar_modelos import cargar_resultados_tradicionales

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            _crear_base_sintetica(raiz)
            base, _, estado = cargar_base(raiz)
            _, asignacion, _ = crear_pliegues_externos(base)
            carpeta = raiz / 'experiments/results/ajuste_modelos'
            carpeta.mkdir(parents=True)
            modelos = ['Referencia_mayoritaria', 'SVM_inicial', 'Random_Forest_inicial',
                       'SVM_ajustado', 'Random_Forest_ajustado']
            tablas = []
            for nombre in modelos:
                tabla = base[['id_ejemplar', 'grupo_particion', 'calidad_revisada', 'fruta', 'fuente']].copy()
                tabla['modelo'], tabla['prediccion'] = nombre, base.calidad_revisada
                tabla['pliegue_validacion'] = asignacion
                tablas.append(tabla)
            predicciones = pd.concat(tablas, ignore_index=True)
            ruta = carpeta / 'predicciones_oof.csv'
            predicciones.to_csv(ruta, index=False)
            (carpeta / 'configuracion_ajuste.json').write_text(json.dumps({'sha_base': estado['sha_base']}), encoding='utf-8')
            (carpeta / 'parametros_seleccionados.json').write_text('[]', encoding='utf-8')
            verificadas, _ = cargar_resultados_tradicionales(raiz, base, estado, asignacion)
            self.assertEqual(set(verificadas), set(modelos))

            modificadas = predicciones.copy()
            modificadas.loc[0, 'grupo_particion'] = 'grupo_ajeno'
            modificadas.to_csv(ruta, index=False)
            with self.assertRaisesRegex(ValueError, 'grupo_particion'):
                cargar_resultados_tradicionales(raiz, base, estado, asignacion)

            modificadas = predicciones.copy()
            modificadas.loc[0, 'pliegue_validacion'] = modificadas.loc[0, 'pliegue_validacion'] % 3 + 1
            modificadas.to_csv(ruta, index=False)
            with self.assertRaisesRegex(ValueError, 'bloques'):
                cargar_resultados_tradicionales(raiz, base, estado, asignacion)

    def test_validacion_agrupada_y_ajuste_tradicional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            _crear_base_sintetica(raiz)
            base, columnas, estado = cargar_base(raiz)
            particiones, asignacion, cobertura = crear_pliegues_externos(base, n_splits=3, semilla=42)

            self.assertEqual(len(particiones), 3)
            self.assertFalse(cobertura.empty)
            for tr, va in particiones:
                self.assertFalse(set(base.iloc[tr].grupo_particion) & set(base.iloc[va].grupo_particion))

            cv_int = crear_cv_interno(base.calidad_revisada, base.grupo_particion)
            self.assertGreaterEqual(cv_int.get_n_splits(), 2)

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            salida = raiz / 'experiments/results'
            ckpts = raiz / 'experiments/checkpoints'
            espacios_rapidos = {
                'SVM_ajustado': (
                    Pipeline([
                        ('escalar', StandardScaler()),
                        ('svc', SVC(probability=True, class_weight='balanced', random_state=42)),
                    ]),
                    {'svc__kernel': ['rbf', 'linear'], 'svc__C': [0.5, 1.0]},
                ),
                'Random_Forest_ajustado': (
                    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1),
                    {'n_estimators': [15, 30], 'max_depth': [None, 4]},
                ),
            }
            oof, resumen = evaluar_tradicionales_anidados(
                base, columnas, particiones, estado, salida, ckpts, semilla=42, espacios=espacios_rapidos
            )
            self.assertIn('SVM_ajustado', oof)
            self.assertIn('Random_Forest_ajustado', oof)
            self.assertTrue((ckpts / 'svm_ajustado.joblib').is_file())
            self.assertTrue((ckpts / 'random_forest_ajustado.joblib').is_file())
            self.assertIn('mejores_parametros_global', resumen['SVM_ajustado'])

            tablas = consolidar_resultados_oof(
                base, particiones, asignacion, oof, salida, generar_graficas=MATPLOTLIB_DISPONIBLE
            )
            self.assertEqual(len(tablas['metricas_globales']), 3)
            self.assertTrue((salida / 'metricas_globales.csv').is_file())
            if MATPLOTLIB_DISPONIBLE:
                self.assertTrue((salida / 'comparacion_modelos.png').is_file())
                self.assertTrue((salida / 'matrices_confusion.png').is_file())

    @unittest.skipUnless(TORCH_DISPONIBLE, 'PyTorch/torchvision no instalados en este entorno.')
    def test_cnn_calidad_y_evaluacion(self) -> None:
        import torch
        from src.models.cnn_calidad import CNNCalidadFruta, ConfigCNN, evaluar_cnn_anidada

        hilos_anteriores = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, hilos_anteriores)
        torch.set_num_threads(min(2, hilos_anteriores))

        modelo = CNNCalidadFruta(num_clases=3, dropout=0.2)
        salida_logits = modelo(torch.zeros((2, 3, 224, 224)))
        self.assertEqual(tuple(salida_logits.shape), (2, 3))

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            _crear_base_sintetica(raiz)
            base, _, estado = cargar_base(raiz)
            particiones, _, _ = crear_pliegues_externos(base, n_splits=3, semilla=42)
            salida = raiz / 'experiments/results'
            ckpts = raiz / 'experiments/checkpoints'
            oof_cnn, info = evaluar_cnn_anidada(
                base,
                particiones,
                proyecto=raiz,
                salida=salida,
                checkpoints=ckpts,
                estado=estado,
                candidatos=[ConfigCNN(lado_entrada=64, max_epocas=2, paciencia=1, batch_size=8)],
                semilla=42,
            )
            self.assertEqual(len(oof_cnn), len(base))
            self.assertTrue(set(oof_cnn) <= set(CALIDADES))
            self.assertTrue((ckpts / 'cnn_calidad.pt').is_file())
            self.assertTrue((salida / 'curvas_entrenamiento_cnn.png').is_file())
            self.assertIn('mejor_config_global', info)


if __name__ == '__main__':
    unittest.main()
