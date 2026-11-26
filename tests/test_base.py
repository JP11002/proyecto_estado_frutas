"""Integridad, rutas y lectura de la única base del proyecto."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import numpy as np
from src.data.base import cargar_archivos_base, resolver_ruta, verificar_imagenes
from src.models.evaluacion import cargar_base


class TestBase(unittest.TestCase):
    def test_rutas_windows_posix_y_relativas(self):
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            esperada = raiz / 'data/base/imagenes/foto.png'
            for texto in ['data/base/imagenes/foto.png',
                          r'C:\Users\otra_persona\frutas\data\base\imagenes\foto.png',
                          '/home/equipo/frutas/data/base/imagenes/foto.png']:
                self.assertEqual(resolver_ruta(raiz, texto), esperada)
            with self.assertRaises(ValueError):
                resolver_ruta(raiz, '../fuera.png')
            with self.assertRaises(ValueError):
                resolver_ruta(raiz, r'C:\otra\carpeta\archivo.png')

    def test_snapshot_relativo_y_sin_cambios_de_etiquetas(self):
        carpeta = RAIZ / 'data/base'
        estado = json.loads((carpeta / 'estado.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256((carpeta / 'datos.csv').read_bytes()).hexdigest(), estado['sha_base'])
        import pandas as pd
        base = pd.read_csv(carpeta / 'datos.csv')
        self.assertEqual(len(base), 291)
        self.assertEqual(len(estado['caracteristicas']), 43)
        for campo in ['ruta_original', 'ruta_cnn', 'ruta_mascara']:
            self.assertFalse(base[campo].str.contains(r'^[A-Za-z]:|^/|\\', regex=True).any())
        self.assertEqual(base.calidad_revisada.value_counts().to_dict(), {'buena': 201, 'intermedia': 75, 'mala': 15})
        self.assertTrue(np.isfinite(base[estado['caracteristicas']].to_numpy()).all())

    def test_carga_fresca_sin_data_processed(self):
        # Redirigir la raíz con enlaces no es necesario: copiar solo los archivos mínimos.
        import shutil
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal) / 'nombre distinto'
            shutil.copytree(RAIZ / 'data/base', raiz / 'data/base')
            base, columnas, _ = cargar_base(raiz)
            for ruta in base.ruta_mascara:
                relativa = Path(ruta).relative_to(raiz)
                destino = raiz / relativa
                destino.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(RAIZ / relativa, destino)
            verificar_imagenes(base)
            self.assertEqual(len(base), 291)
            self.assertEqual(len(columnas), 43)
            self.assertFalse((raiz / 'data/processed').exists())
            self.assertTrue(all(Path(p).is_relative_to(raiz) for p in base.ruta_cnn))
            # Una tabla temporal nunca sustituye la base oficial.
            (raiz / 'data/processed').mkdir()
            (raiz / 'data/processed/datos.csv').write_text('tabla_temporal\nincorrecta\n', encoding='utf-8')
            recargada, _, _ = cargar_base(raiz)
            self.assertEqual(recargada.id_ejemplar.tolist(), base.id_ejemplar.tolist())


if __name__ == '__main__':
    unittest.main()
