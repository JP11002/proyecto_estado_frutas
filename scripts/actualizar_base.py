"""Actualizar data/base/ a partir de etiquetas, máscaras y originales revisados."""
import argparse
from pathlib import Path
import sys

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.data.exportar_base import exportar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proyecto', default=str(PROYECTO))
    base, resumen, _ = exportar(parser.parse_args().proyecto)
    print(resumen.to_string(index=False))
    print(f'Base actualizada: {len(base)} registros en data/base/datos.csv.')
