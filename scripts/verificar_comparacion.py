"""Verificar resultados OOF, separación de grupos y recarga de pesos de la CNN."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))

from src.data.base import cargar_base, verificar_imagenes
from src.models.cnn_calidad import (CALIDADES, CNNCalidadFruta, predecir_rutas_cnn,
                                  CNNTransferencia, ConfigTransferencia, predecir_transferencia)
from src.models.evaluacion import calcular_metricas, crear_pliegues_externos


def verificar(proyecto=PROYECTO):
    proyecto = Path(proyecto).resolve()
    salida = proyecto / 'experiments/results/comparacion_modelos'
    checkpoints = proyecto / 'experiments/checkpoints/comparacion_modelos'
    base, _, estado = cargar_base(proyecto)
    verificar_imagenes(base)
    metadata = json.loads((salida / 'configuracion_comparacion.json').read_text(encoding='utf-8'))
    assert metadata['sha_base'] == estado['sha_base']
    assert metadata['evaluacion_independiente'] is False
    torch.set_num_threads(metadata['resumen_cnn']['hilos_cpu'])
    particiones, asignacion, _ = crear_pliegues_externos(base)
    predicciones = pd.read_csv(salida / 'predicciones_oof.csv')
    metricas = pd.read_csv(salida / 'metricas_globales.csv').set_index('modelo')
    modelos = metadata['modelos_evaluados']
    assert set(predicciones.modelo) == set(modelos)
    assert len(predicciones) == len(base) * len(modelos)
    assert not predicciones[['modelo', 'id_ejemplar']].duplicated().any()
    for nombre, sub in predicciones.groupby('modelo', sort=False):
        assert sub.id_ejemplar.tolist() == base.id_ejemplar.tolist()
        assert sub.calidad_revisada.tolist() == base.calidad_revisada.tolist()
        assert sub.grupo_particion.tolist() == base.grupo_particion.tolist()
        assert sub.pliegue_validacion.tolist() == asignacion.tolist()
        assert np.array_equal(sub.acierto, sub.calidad_revisada.eq(sub.prediccion))
        for metrica, valor in calcular_metricas(sub.calidad_revisada, sub.prediccion).items():
            assert np.isclose(valor, metricas.loc[nombre, metrica])

    tradicionales = pd.read_csv(proyecto / 'experiments/results/ajuste_modelos/predicciones_oof.csv')
    for nombre, sub in tradicionales.groupby('modelo'):
        esperado = sub.set_index('id_ejemplar').loc[base.id_ejemplar].prediccion.tolist()
        assert predicciones.loc[predicciones.modelo.eq(nombre), 'prediccion'].tolist() == esperado

    auditoria = pd.read_csv(salida / 'cnn_particiones.csv')
    oof = pd.read_csv(salida / 'cnn_predicciones_oof.csv')
    assert oof.id_ejemplar.tolist() == base.id_ejemplar.tolist()
    assert oof.pliegue_validacion.tolist() == asignacion.tolist()
    nombre_cnn = 'CNN_transferencia' if 'CNN_transferencia' in modelos else 'CNN_simple'
    assert predicciones.loc[predicciones.modelo.eq(nombre_cnn), 'prediccion'].tolist() == oof.prediccion.tolist()
    if 'CNN_transferencia' in modelos and 'CNN_simple' in modelos:
        referencia = pd.read_csv(salida / 'cnn_referencia_anterior.csv').set_index('id_ejemplar').loc[base.id_ejemplar]
        assert predicciones.loc[predicciones.modelo.eq('CNN_simple'), 'prediccion'].tolist() == referencia.prediccion.tolist()
    probabilidades = oof[[f'prob_{c}' for c in CALIDADES]].to_numpy()
    assert np.isfinite(probabilidades).all()
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
    assert np.allclose(probabilidades.sum(axis=1), 1, atol=1e-6)
    assert oof.prediccion.tolist() == [CALIDADES[i] for i in probabilidades.argmax(axis=1)]

    historial = pd.read_csv(salida / 'cnn_historial_epocas.csv')
    assert 'loss_val' not in historial and 'f1_val' not in historial
    configuraciones = pd.read_csv(salida / 'cnn_configuraciones_externas.csv').set_index('exterior')
    plan = json.loads((salida / 'cnn_plan_entrenamiento.json').read_text(encoding='utf-8'))
    busqueda = pd.read_csv(salida / 'cnn_busqueda_interna.csv')
    for exterior, (train, val) in enumerate(particiones, 1):
        bloque = auditoria.loc[auditoria.exterior.eq(exterior)]
        exterior_train = bloque.query("interior == 0 and etapa == 'entrenamiento'")
        exterior_val = bloque.query("interior == 0 and etapa == 'validacion'")
        assert set(exterior_train.id_ejemplar) == set(base.iloc[train].id_ejemplar)
        assert set(exterior_val.id_ejemplar) == set(base.iloc[val].id_ejemplar)
        assert not set(exterior_train.grupo) & set(exterior_val.grupo)
        for interior in sorted(set(bloque.interior) - {0}):
            sub = bloque.loc[bloque.interior.eq(interior)]
            tr = sub.loc[sub.etapa.eq('entrenamiento')]
            va = sub.loc[sub.etapa.eq('validacion')]
            assert not set(tr.grupo) & set(va.grupo)
            assert set(tr.id_ejemplar) | set(va.id_ejemplar) == set(base.iloc[train].id_ejemplar)
            assert not set(sub.grupo) & set(exterior_val.grupo)
        candidatos = busqueda.loc[busqueda.pliegue_externo.eq(exterior)]
        assert len(candidatos) == len(plan['candidatos'])
        mejor = candidatos.iloc[0]
        cfg = configuraciones.loc[exterior]
        for campo in ['lr', 'dropout', 'weight_decay']:
            assert np.isclose(cfg[campo], mejor[campo])
        if 'potencia_peso' in cfg:
            assert cfg.cabezal == mejor.cabezal
            assert cfg.potencia_peso == mejor.potencia_peso
        minimo = int(mejor.get('min_epocas', 1))
        assert cfg.max_epocas == max(minimo, min(int(mejor.max_epocas), int(mejor.epoca_media)))
        assert candidatos.f1_macro_interno_medio.is_monotonic_decreasing
        assert len(historial.loc[historial.pliegue.eq(exterior)]) == cfg.max_epocas

        paquete = torch.load(checkpoints / f'cnn_pliegue_{exterior}.pt', map_location='cpu', weights_only=True)
        assert paquete['sha_base'] == estado['sha_base']
        assert paquete['ids_train'] == base.iloc[train].id_ejemplar.tolist()
        assert paquete['ids_validacion'] == base.iloc[val].id_ejemplar.tolist()
        transferencia = paquete.get('arquitectura') == 'mobilenet_v3_small_congelada'
        modelo = CNNTransferencia(ConfigTransferencia(**paquete['config'])) if transferencia else CNNCalidadFruta(dropout=paquete['config']['dropout'])
        modelo.load_state_dict(paquete['state_dict'])
        if transferencia:
            pred, probs = predecir_transferencia(modelo, [Path(p) for p in base.iloc[val].ruta_cnn])
            assert not any(p.requires_grad for p in modelo.extractor.parameters())
        else:
            pred, probs = predecir_rutas_cnn(modelo, [Path(p) for p in base.iloc[val].ruta_cnn],
                                            lado_entrada=paquete['config']['lado_entrada'])
        assert pred.tolist() == oof.iloc[val].prediccion.tolist()
        assert np.allclose(probs, probabilidades[val], atol=1e-6)

    final = torch.load(checkpoints / 'cnn_calidad.pt', map_location='cpu', weights_only=True)
    assert final['sha_base'] == estado['sha_base']
    assert final['calidades'] == CALIDADES
    if final.get('arquitectura') == 'mobilenet_v3_small_congelada':
        globales = pd.read_csv(salida / 'cnn_busqueda_global.csv')
        assert globales.f1_macro_interno_medio.is_monotonic_decreasing
        mejor = globales.iloc[0]
        for campo in ['cabezal', 'potencia_peso', 'lr', 'dropout', 'weight_decay']:
            assert final['config'][campo] == mejor[campo]
        assert final['config']['max_epocas'] == max(int(mejor.min_epocas), min(int(mejor.max_epocas), int(mejor.epoca_media)))
        global_auditoria = auditoria.loc[auditoria.exterior.eq(0)]
        assert global_auditoria.interior.nunique() == 3
        for interior, sub in global_auditoria.groupby('interior'):
            tr = sub.loc[sub.etapa.eq('entrenamiento')]
            va = sub.loc[sub.etapa.eq('validacion')]
            assert not set(tr.grupo) & set(va.grupo)
            assert set(tr.id_ejemplar) | set(va.id_ejemplar) == set(base.id_ejemplar)
    transferencia = final.get('arquitectura') == 'mobilenet_v3_small_congelada'
    modelo = CNNTransferencia(ConfigTransferencia(**final['config'])) if transferencia else CNNCalidadFruta(dropout=final['config']['dropout'])
    modelo.load_state_dict(final['state_dict'])
    pred, probs = (predecir_transferencia if transferencia else predecir_rutas_cnn)(modelo, [Path(base.iloc[0].ruta_cnn)])
    assert len(pred) == 1 and np.isfinite(probs).all()
    historial_final = pd.read_csv(salida / 'cnn_historial_final.csv')
    assert len(historial_final) == final['config']['max_epocas'] == metadata['resumen_cnn']['epocas_finales']
    print(f'Verificado: {len(predicciones)} predicciones OOF, {len(modelos)} modelos y {len(base)} imágenes por modelo.')
    print('Separación de grupos, selección interna, métricas y probabilidades correctas.')
    print('Los pesos de los tres bloques reproducen sus predicciones externas; la CNN final se recarga correctamente.')


if __name__ == '__main__':
    verificar()
