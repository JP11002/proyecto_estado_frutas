"""Comprobar una copia del proyecto sin originales ni modelos entrenados."""
from __future__ import annotations

import argparse
import importlib
import platform
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cnn', action='store_true', help='Comprobar también PyTorch y torchvision.')
    args = parser.parse_args()
    proyecto = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(proyecto))
    print('Python:', platform.python_version())
    if sys.version_info < (3, 11):
        print('Se requiere Python >=3.11. Se recomienda 3.11 para reproducir el entorno probado.')
        return 1
    requeridos = ['numpy', 'pandas', 'matplotlib', 'sklearn', 'joblib', 'PIL',
                  'cv2', 'nbformat', 'nbclient', 'ipykernel', 'ipywidgets']
    if args.cnn:
        requeridos += ['torch', 'torchvision']
    fallos = []
    for nombre in requeridos:
        try:
            modulo = importlib.import_module(nombre)
            print(f'OK {nombre}: {getattr(modulo, "__version__", "disponible")}')
        except (ImportError, OSError, RuntimeError) as error:
            fallos.append(f'{nombre}: {error}')
    if fallos:
        print('\nDependencias pendientes:\n' + '\n'.join(fallos))
        archivo = 'requirements-dl.txt' if args.cnn else 'requirements.txt'
        print(f'Instalar en este mismo entorno: python -m pip install -r {archivo}')
        return 1
    try:
        from src.models.evaluacion import cargar_base
        from src.data.base import verificar_imagenes
        base, columnas, _ = cargar_base(proyecto)
        verificar_imagenes(base)
        print(f'OK base: {len(base)} imágenes, {len(columnas)} características y {base.grupo_particion.nunique()} grupos.')
        print('OK imágenes preparadas y máscaras: sus huellas coinciden. No hacen falta fotos originales para entrenar.')
        if args.cnn:
            from src.models.cnn_calidad import CNNTransferencia, ConfigTransferencia
            import torch
            torch.set_num_threads(min(2, torch.get_num_threads()))
            modelo = CNNTransferencia(ConfigTransferencia()).eval()
            with torch.no_grad():
                assert tuple(modelo(torch.zeros(1, 3, 224, 224)).shape) == (1, 3)
            print('OK CNN MobileNetV3: arquitectura comprobada en CPU sin descargar pesos. No se requiere GPU.')
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        print('ERROR:', error)
        return 1
    print('Proyecto listo para ejecutar el EDA y los entrenamientos.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
