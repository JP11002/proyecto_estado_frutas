import unittest
import numpy as np
from src.inference.motor import medir_tamano

class TestTamano(unittest.TestCase):
    def test_invariante_a_resolucion(self):
        m=np.zeros((100,200),dtype=np.uint8);m[20:80,60:140]=1
        a=medir_tamano(m);b=medir_tamano(np.repeat(np.repeat(m,2,0),2,1))
        self.assertAlmostEqual(a['diametro_normalizado'],b['diametro_normalizado'])
        self.assertAlmostEqual(a['ocupacion'],b['ocupacion'])
        self.assertAlmostEqual(2*a['diametro_equivalente_px'],b['diametro_equivalente_px'])
    def test_rechaza_mascara_vacia(self):
        with self.assertRaises(ValueError):medir_tamano(np.zeros((100,100)))
    def test_encuadre_importa(self):
        a=np.zeros((100,100));a[20:80,20:80]=1
        b=np.zeros((200,200));b[20:80,20:80]=1
        self.assertAlmostEqual(medir_tamano(a)['diametro_normalizado'],2*medir_tamano(b)['diametro_normalizado'])


class TestFotografiaCompleta(unittest.TestCase):
    def test_conserva_encuadre_y_proporciones(self):
        from src.data.entrada_cnn import preparar_imagen_completa
        rgb=np.full((100,200,3),240,dtype=np.uint8)
        rgb[:10,:10]=[255,0,0];rgb[-10:,-10:]=[0,0,255]
        salida=preparar_imagen_completa(rgb,lado=200)
        self.assertEqual(salida.shape,(200,200,3))
        self.assertTrue(np.array_equal(salida[50,0],[255,0,0]))
        self.assertTrue(np.array_equal(salida[149,199],[0,0,255]))
        self.assertTrue(np.all(salida[:50]==128))
        self.assertTrue(np.array_equal(salida[50:150],rgb))

    def test_cnn_clasifica_aunque_falle_la_mascara(self):
        from unittest.mock import patch
        from src.inference.motor import MotorInferencia
        motor=MotorInferencia.__new__(MotorInferencia)
        motor._predecir=lambda entrada,rasgos,nombre: {'calidad':'buena'}
        with patch('src.inference.motor.mascara_grabcut',side_effect=ValueError('Sin contorno')):
            resultado,entrada,mascara=motor.analizar(np.full((80,120,3),200,dtype=np.uint8))
        self.assertEqual(resultado['calidad'],'buena')
        self.assertIsNone(resultado['tamano'])
        self.assertIsNone(mascara)
        self.assertEqual(entrada.shape,(224,224,3))
