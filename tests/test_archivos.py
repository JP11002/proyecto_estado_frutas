"""El guardado tolera bloqueos breves de lectura sin perder datos."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.archivos import guardar_bytes_si_cambio


class TestArchivos(unittest.TestCase):
    def test_reintento_de_reemplazo_bloqueado(self):
        with tempfile.TemporaryDirectory() as temporal:
            destino = Path(temporal) / 'avance.json'
            destino.write_bytes(b'contenido anterior')
            original = Path.replace
            intentos = []

            def reemplazar(origen, ruta):
                intentos.append(1)
                if len(intentos) <= 2:
                    raise PermissionError('Bloqueo temporal de prueba')
                return original(origen, ruta)

            with patch.object(Path, 'replace', reemplazar), patch('src.utils.archivos.time.sleep'):
                guardar_bytes_si_cambio(destino, b'contenido nuevo', atomico=True)
            self.assertEqual(destino.read_bytes(), b'contenido nuevo')
            self.assertEqual(len(intentos), 3)
            self.assertFalse(list(Path(temporal).glob('*.tmp')))

    def test_bloqueo_permanente_conserva_el_anterior(self):
        with tempfile.TemporaryDirectory() as temporal:
            destino = Path(temporal) / 'avance.json'
            destino.write_bytes(b'contenido anterior')
            with patch.object(Path, 'replace', side_effect=PermissionError('Bloqueado')), patch('src.utils.archivos.time.sleep'):
                with self.assertRaises(PermissionError):
                    guardar_bytes_si_cambio(destino, b'contenido nuevo', atomico=True)
            self.assertEqual(destino.read_bytes(), b'contenido anterior')

    def test_escritura_directa_no_renombra_archivos(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta=Path(carpeta)/'resultado.json';ruta.write_bytes(b'anterior')
            with patch.object(Path,'replace',side_effect=AssertionError('No debe renombrar')):
                guardar_bytes_si_cambio(ruta,b'nuevo',atomico=False)
            self.assertEqual(ruta.read_bytes(),b'nuevo')

    def test_bloqueo_directo_conserva_archivo(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta=Path(carpeta)/'resultado.json';ruta.write_bytes(b'anterior')
            with patch.object(Path,'write_bytes',side_effect=PermissionError('Bloqueado')), patch('src.utils.archivos.time.sleep'):
                with self.assertRaises(PermissionError):guardar_bytes_si_cambio(ruta,b'nuevo',atomico=False)
            self.assertEqual(ruta.read_bytes(),b'anterior')


if __name__ == '__main__':
    unittest.main()
