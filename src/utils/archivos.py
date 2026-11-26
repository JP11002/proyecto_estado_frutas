"""Operaciones idempotentes de lectura y guardado de archivos."""
from __future__ import annotations

from pathlib import Path
import time
import os
import tempfile


def guardar_bytes_si_cambio(ruta: str | Path, contenido: bytes, *, atomico: bool | None = None) -> None:
    """Reutilizar archivos idénticos para evitar reescrituras innecesarias."""
    ruta = Path(ruta)
    if ruta.is_file():
        anterior = ruta.read_bytes()
        if anterior == contenido:
            return
        if ruta.suffix in {'.json', '.csv'} and anterior.replace(b'\r\n', b'\n') == contenido.replace(b'\r\n', b'\n'):
            return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if atomico is None:
        atomico = os.name != 'nt'
    if not atomico:
        # Evitar renombrados de reemplazo en carpetas sincronizadas de Windows.
        # Los lectores verifican las huellas para rechazar una escritura incompleta.
        respaldo = ruta.read_bytes() if ruta.is_file() else None
        try:
            for intento in range(8):
                try:
                    ruta.write_bytes(contenido)
                    return
                except PermissionError:
                    if intento == 7:
                        raise
                    time.sleep(min(.05 * 2**intento, .5))
        except BaseException:
            if respaldo is not None:
                try:
                    ruta.write_bytes(respaldo)
                except OSError:
                    pass
            raise
    descriptor, nombre = tempfile.mkstemp(dir=ruta.parent, suffix='.tmp')
    temporal = Path(nombre)
    with os.fdopen(descriptor, 'wb') as archivo:
        archivo.write(contenido)
    try:
        # Windows puede bloquear brevemente el destino mientras lo lee otra app.
        for intento in range(8):
            try:
                temporal.replace(ruta)
                break
            except PermissionError:
                if intento == 7:
                    raise
                time.sleep(min(.05 * 2**intento, .5))
    finally:
        try:
            temporal.unlink(missing_ok=True)
        except PermissionError:
            pass


def guardar_tabla(tabla, ruta: str | Path) -> None:
    """Guardar DataFrame como CSV UTF-8 con BOM solo si cambió."""
    guardar_bytes_si_cambio(ruta, tabla.to_csv(index=False).encode('utf-8-sig'))
