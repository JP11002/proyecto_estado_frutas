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
