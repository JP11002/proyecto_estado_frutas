"""Empaquetar el mismo proyecto, base, modelos y evidencias para la demostración."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import subprocess,json,hashlib


def main():
    p=Path(__file__).resolve().parents[1]
    salida=p/'output/entrega';salida.mkdir(parents=True,exist_ok=True)
    lista=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=p).decode('utf-8').split('\0')
    archivos={p/x for x in lista if x and (p/x).is_file()}
    for carpeta in ['experiments/checkpoints','experiments/results']:
        archivos.update(f for f in (p/carpeta).rglob('*') if f.is_file() and f.suffix.lower() in {'.pt','.joblib','.csv','.json','.png'} and f.name!='cnn_caracteristicas.pt')
    # Este paquete no contiene originales de Drive, cachés, entornos ni credenciales.
    cfg=json.loads((p/'config/entrega.json').read_text(encoding='utf-8'))
    manifest={'proyecto':'proyecto_estado_frutas','estado':'pendiente_enlace_video' if not cfg['video'] else 'con_enlace_video',
              'repositorio':cfg['repositorio'],'video':cfg['video'],'integrantes':cfg['integrantes'],
              'sha_base':hashlib.sha256((p/'data/base/datos.csv').read_bytes()).hexdigest(),
              'archivos':len(archivos),'nota':'Paquete del mismo proyecto con modelos; no reemplaza la publicación del repositorio ni del video.'}
    destino=salida/'proyecto_estado_frutas.zip'
    with ZipFile(destino,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(archivos):z.write(f,'proyecto_estado_frutas/'+f.relative_to(p).as_posix())
        z.writestr('proyecto_estado_frutas/manifest_entrega.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    (salida/'manifest_entrega.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Paquete: {destino} | {destino.stat().st_size/1024**2:.1f} MB | {manifest["estado"]}')
if __name__=='__main__':main()
