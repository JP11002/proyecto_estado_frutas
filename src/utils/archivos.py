"""Operaciones idempotentes de lectura y guardado de archivos."""
from __future__ import annotations

from pathlib import Path


def guardar_bytes_si_cambio(ruta: str | Path, contenido: bytes) -> None:
    """Reutilizar archivos idénticos para evitar reescrituras innecesarias."""
    ruta = Path(ruta)
    if ruta.is_file():
        anterior = ruta.read_bytes()
        if anterior == contenido:
            return
        if ruta.suffix in {'.json', '.csv'} and anterior.replace(b'\r\n', b'\n') == contenido.replace(b'\r\n', b'\n'):
            return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + '.tmp')
    temporal.write_bytes(contenido)
    temporal.replace(ruta)


def guardar_tabla(tabla, ruta: str | Path) -> None:
    """Guardar DataFrame como CSV UTF-8 con BOM solo si cambió."""
    guardar_bytes_si_cambio(ruta, tabla.to_csv(index=False).encode('utf-8-sig'))
