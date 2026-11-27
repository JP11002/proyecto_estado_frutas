"""Ejecutar un cuaderno con el Python actual, sin kernels de otro computador."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cuaderno')
    parser.add_argument('--timeout', type=int, default=3600, help='Máximo de segundos por celda.')
    args = parser.parse_args()
    proyecto = Path(__file__).resolve().parents[1]
    ruta = Path(args.cuaderno)
    if not ruta.is_absolute():
        ruta = proyecto / ruta
    ruta = ruta.resolve()
    if not ruta.is_relative_to(proyecto / 'notebooks') or ruta.suffix != '.ipynb':
        raise ValueError('Indicar un cuaderno de la carpeta notebooks del proyecto.')
    nb = nbformat.read(ruta, as_version=4)
    with TemporaryDirectory(prefix='frutas_kernel_') as temporal:
        kernels = Path(temporal) / 'kernels'
        carpeta = kernels / 'frutas_actual'
        carpeta.mkdir(parents=True)
        (carpeta / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python actual (frutas)', 'language': 'python',
            'env': {'MPLBACKEND': 'module://matplotlib_inline.backend_inline'},
        }), encoding='utf-8')
        gestor = KernelManager(kernel_name='frutas_actual',
                               kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        cliente = NotebookClient(nb, km=gestor, timeout=args.timeout,
                                 resources={'metadata': {'path': str(ruta.parent)}})
        cliente.on_cell_start = lambda cell, cell_index, **kw: print(f'Celda {cell_index}', flush=True)
        try:
            cliente.execute()
        finally:
            if gestor.has_kernel:
                gestor.shutdown_kernel(now=True)
    # El kernel temporal desaparece; conservar una selección estándar al abrirlo.
    nb.metadata['kernelspec'] = {'display_name': 'Python 3 (frutas)', 'language': 'python', 'name': 'python3'}
    nbformat.write(nb, ruta)
    print('Cuaderno ejecutado y guardado:', ruta)


if __name__ == '__main__':
    main()
