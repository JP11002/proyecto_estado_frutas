"""Entrada: python -m src.main (interfaz) o --ejemplar ID / --imagen RUTA (terminal)."""
import argparse,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    entrada=parser.add_mutually_exclusive_group()
    entrada.add_argument('--imagen');entrada.add_argument('--ejemplar')
    parser.add_argument('--modelo',default='CNN',help='Uno de MotorInferencia.disponibles(): CNN, Random Forest, SVM o el modelo final v2.')
    parser.add_argument('--salida',type=Path)
    args=parser.parse_args()
    if not args.imagen and not args.ejemplar:
        from src.interfaz import iniciar
        iniciar();return
    from src.inference.motor import MotorInferencia
    from src.data.preparacion_imagenes import leer_rgb
    motor=MotorInferencia()
    resultado,_,_=motor.ejemplo(args.ejemplar,args.modelo) if args.ejemplar else motor.analizar(leer_rgb(args.imagen),args.modelo)
    texto=json.dumps(resultado,ensure_ascii=False,indent=2)
    print(texto)
    if args.salida:args.salida.write_text(texto,encoding='utf-8')


if __name__=='__main__':main()
