"""CNN de calidad sobre la base común: transferencia con MobileNetV3 congelada.

Se conserva la CNN simple como referencia del experimento anterior. La entrada
son los recortes RGB preparados de 224x224 y la evaluación mantiene los frutos
y campañas separados por grupos.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.data.base import resolver_ruta

from src.models.evaluacion import (
    CALIDADES,
    SEMILLA,
    crear_cv_interno,
    guardar_grafica,
    guardar_json,
)
from src.utils.archivos import guardar_tabla

CLASE_A_INDICE = {c: i for i, c in enumerate(CALIDADES)}
INDICE_A_CLASE = {i: c for i, c in enumerate(CALIDADES)}
MEDIA_NORM = (0.5, 0.5, 0.5)
DESV_NORM = (0.5, 0.5, 0.5)


@lru_cache(maxsize=512)
def _imagen_rgb_en_memoria(ruta: str, modificacion: int) -> Image.Image:
    """Cachear solo píxeles sin transformar; los aumentos siguen siendo aleatorios."""
    with Image.open(ruta) as imagen:
        return imagen.convert('RGB')


@dataclass(frozen=True)
class ConfigCNN:
    """Hiperparámetros entrenables y de arquitectura de la CNN simple."""

    lado_entrada: int = 224
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_epocas: int = 25
    paciencia: int = 7


def fijar_semillas(semilla: int = SEMILLA) -> None:
    """Fijar semillas de Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def resolver_ruta_cnn(fila: pd.Series, proyecto: Path) -> Path:
    """Leer la imagen indicada por la base oficial, sin buscar copias alternativas."""
    ruta = resolver_ruta(proyecto, str(fila['ruta_cnn']))
    if ruta.is_file():
        return ruta
    raise FileNotFoundError(f'Falta la imagen de {fila["id_ejemplar"]} en la base: {ruta}.')


def construir_transformaciones(lado: int = 224, entrenamiento: bool = True) -> transforms.Compose:
    """Transformaciones de imagen; nunca altera tono bruscamente ni aplica aumento en evaluación."""
    pasos: list[Any] = [transforms.Resize((lado, lado))]
    if entrenamiento:
        pasos.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(degrees=15, fill=128),
            # Brillo/contraste moderados; evitamos cambios fuertes de tono (hue) que alteran madurez.
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
        ])
    pasos.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEDIA_NORM, std=DESV_NORM),
    ])
    return transforms.Compose(pasos)


class DatasetFrutasCNN(Dataset):
    """Dataset PyTorch sobre recortes RGB preparados y etiquetas de calidad."""

    def __init__(
        self,
        rutas: list[Path],
        etiquetas: list[int] | None = None,
        transformacion: transforms.Compose | None = None,
    ) -> None:
        self.rutas = [Path(r) for r in rutas]
        self.etiquetas = etiquetas
        self.transformacion = transformacion or construir_transformaciones(entrenamiento=False)

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, idx: int):
        ruta = self.rutas[idx]
        rgb = _imagen_rgb_en_memoria(str(ruta), ruta.stat().st_mtime_ns).copy()
        tensor = self.transformacion(rgb)
        if self.etiquetas is None:
            return tensor
        return tensor, int(self.etiquetas[idx])


class CNNCalidadFruta(nn.Module):
    """CNN simple desde cero: 3 bloques convolucionales + pooling + cabezal denso."""

    def __init__(self, num_clases: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clasificador(self.extractor(x))


def calcular_pesos_clase(etiquetas: np.ndarray, num_clases: int = 3) -> torch.Tensor:
    """Calcular pesos inversamente proporcionales a la frecuencia del entrenamiento."""
    conteos = np.bincount(etiquetas, minlength=num_clases).astype(float)
    conteos = np.maximum(conteos, 1.0)
    pesos = len(etiquetas) / (num_clases * conteos)
    return torch.tensor(pesos, dtype=torch.float32)


def _evaluar_loader(
    modelo: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    modelo.eval()
    perdida_total = 0.0
    total = 0
    reales, predichas, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logits = modelo(x)
            perdida = criterio(logits, y)
            peso_batch = float(criterio.weight[y].sum().item()) if criterio.weight is not None else len(y)
            perdida_total += float(perdida.item()) * peso_batch
            total += peso_batch
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(prob)
            reales.extend(y.cpu().numpy().tolist())
            predichas.extend(prob.argmax(axis=1).tolist())
    reales_arr = np.asarray(reales)
    pred_arr = np.asarray(predichas)
    f1 = float(f1_score(reales_arr, pred_arr, labels=list(range(len(CALIDADES))), average='macro', zero_division=0))
    return perdida_total / max(total, 1), f1, pred_arr, np.vstack(probs)


def entrenar_un_ajuste_cnn(
    rutas_train: list[Path],
    y_train: np.ndarray,
    rutas_val: list[Path] | None,
    y_val: np.ndarray | None,
    config: ConfigCNN,
    semilla: int = SEMILLA,
    usar_val_para_early_stopping: bool = True,
    progreso=None,
) -> tuple[CNNCalidadFruta, list[dict[str, float]], dict[str, Any]]:
    """Entrenar una instancia de CNNCalidadFruta.

    Cuando `usar_val_para_early_stopping=False`, el conjunto de validación solo se
    registra como diagnóstico pasivo por época sin influir en la selección de pesos.
    """
    fijar_semillas(semilla)
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_train = DatasetFrutasCNN(
        rutas_train,
        y_train.tolist(),
        construir_transformaciones(config.lado_entrada, entrenamiento=True),
    )
    loader_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, num_workers=0)

    loader_val = None
    if rutas_val is not None and y_val is not None and len(rutas_val) > 0:
        ds_val = DatasetFrutasCNN(
            rutas_val,
            y_val.tolist(),
            construir_transformaciones(config.lado_entrada, entrenamiento=False),
        )
        loader_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False, num_workers=0)

    modelo = CNNCalidadFruta(num_clases=len(CALIDADES), dropout=config.dropout).to(dispositivo)
    pesos = calcular_pesos_clase(y_train, len(CALIDADES)).to(dispositivo)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    historial: list[dict[str, float]] = []
    mejor_f1 = -1.0
    mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    mejor_epoca = 0
    sin_mejora = 0
    control_early_stopping = loader_val is not None and usar_val_para_early_stopping

    for epoca in range(1, config.max_epocas + 1):
        modelo.train()
        perdida_sum = 0.0
        n_muestras = 0
        preds_tr, real_tr = [], []
        for x, y in loader_train:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optimizador.zero_grad()
            logits = modelo(x)
            perdida = criterio(logits, y)
            perdida.backward()
            optimizador.step()
            peso_batch = float(pesos[y].sum().item())
            perdida_sum += float(perdida.item()) * peso_batch
            n_muestras += peso_batch
            preds_tr.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            real_tr.extend(y.detach().cpu().numpy().tolist())

        loss_train = perdida_sum / max(n_muestras, 1)
        f1_train = float(f1_score(real_tr, preds_tr, labels=list(range(len(CALIDADES))), average='macro', zero_division=0))
        registro = {'epoca': epoca, 'loss_train': loss_train, 'f1_train': f1_train}

        if loader_val is not None:
            loss_val, f1_val, _, _ = _evaluar_loader(modelo, loader_val, criterio, dispositivo)
            registro.update({'loss_val': loss_val, 'f1_val': f1_val})
            if control_early_stopping:
                if f1_val > mejor_f1:
                    mejor_f1 = f1_val
                    mejor_epoca = epoca
                    mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
                    sin_mejora = 0
                else:
                    sin_mejora += 1
            else:
                mejor_epoca = epoca
                mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            mejor_epoca = epoca
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}

        historial.append(registro)
        if progreso is not None:
            progreso(registro)
        if control_early_stopping and sin_mejora >= config.paciencia:
            break

    modelo.load_state_dict(mejor_estado)
    resumen = {'mejor_epoca': mejor_epoca, 'mejor_f1_val': mejor_f1, 'epocas_ejecutadas': len(historial)}
    return modelo, historial, resumen


def predecir_rutas_cnn(
    modelo: CNNCalidadFruta,
    rutas: list[Path],
    lado_entrada: int = 224,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Predecir etiquetas textuales y probabilidades sobre una lista de imágenes 224x224."""
    dispositivo = next(modelo.parameters()).device
    modelo.eval()
    ds = DatasetFrutasCNN(rutas, None, construir_transformaciones(lado_entrada, entrenamiento=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    with torch.no_grad():
        for x in loader:
            logits = modelo(x.to(dispositivo))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    matriz_probs = np.vstack(probs)
    etiquetas = np.array([INDICE_A_CLASE[int(i)] for i in matriz_probs.argmax(axis=1)], dtype=object)
    return etiquetas, matriz_probs


def rejilla_cnn_por_defecto(max_epocas: int = 25) -> list[ConfigCNN]:
    """Rejilla compacta y disciplinada para búsqueda de hiperparámetros de la CNN."""
    return [
        ConfigCNN(lr=1e-3, dropout=0.2, weight_decay=1e-4, max_epocas=max_epocas),
        ConfigCNN(lr=1e-3, dropout=0.4, weight_decay=5e-4, max_epocas=max_epocas),
        ConfigCNN(lr=3e-4, dropout=0.2, weight_decay=1e-4, max_epocas=max_epocas),
        ConfigCNN(lr=3e-4, dropout=0.4, weight_decay=5e-4, max_epocas=max_epocas),
    ]


def seleccionar_hiperparametros_cnn(
    rutas: list[Path],
    y_idx: np.ndarray,
    grupos: np.ndarray,
    candidatos: list[ConfigCNN],
    semilla: int = SEMILLA,
    progreso=None,
) -> tuple[ConfigCNN, pd.DataFrame]:
    """Buscar mejores hiperparámetros para la CNN mediante validación interna agrupada."""
    cv_interno = crear_cv_interno(y_idx, grupos, max_splits=3, semilla=semilla)
    splits_internos = list(cv_interno.split(np.zeros(len(y_idx)), y_idx, grupos))
    registros = []

    for idx_cfg, cfg in enumerate(candidatos):
        f1_pliegues = []
        epocas_opt = []
        for idx_sub, (tr_sub, va_sub) in enumerate(splits_internos):
            if set(grupos[tr_sub]) & set(grupos[va_sub]):
                raise ValueError('Un grupo apareció a ambos lados de la selección interna.')
            if set(y_idx[tr_sub]) != set(range(len(CALIDADES))):
                raise ValueError('El entrenamiento interno no contiene las tres calidades.')
            rutas_tr = [rutas[i] for i in tr_sub]
            rutas_va = [rutas[i] for i in va_sub]
            _, _, info = entrenar_un_ajuste_cnn(
                rutas_tr,
                y_idx[tr_sub],
                rutas_va,
                y_idx[va_sub],
                cfg,
                semilla=semilla + idx_sub,
                progreso=(lambda registro, c=idx_cfg+1, p=idx_sub+1:
                          progreso({'candidato': c, 'interior': p, **registro})) if progreso else None,
            )
            f1_pliegues.append(info['mejor_f1_val'])
            epocas_opt.append(info['mejor_epoca'])
        registros.append({
            **asdict(cfg),
            'f1_macro_interno_medio': float(np.mean(f1_pliegues)),
            'f1_macro_interno_std': float(np.std(f1_pliegues)),
            'epoca_media': int(round(np.mean(epocas_opt))),
        })

    tabla = pd.DataFrame(registros).sort_values('f1_macro_interno_medio', ascending=False).reset_index(drop=True)
    mejor_fila = tabla.iloc[0]
    mejor_cfg = ConfigCNN(
        lado_entrada=int(mejor_fila['lado_entrada']),
        dropout=float(mejor_fila['dropout']),
        lr=float(mejor_fila['lr']),
        weight_decay=float(mejor_fila['weight_decay']),
        batch_size=int(mejor_fila['batch_size']),
        max_epocas=max(1, min(int(mejor_fila['max_epocas']), int(mejor_fila['epoca_media']))),
        paciencia=int(mejor_fila['paciencia']),
    )
    return mejor_cfg, tabla


def evaluar_cnn_anidada(
    base: pd.DataFrame,
    particiones: list[tuple[np.ndarray, np.ndarray]],
    proyecto: str | Path,
    salida: str | Path,
    checkpoints: str | Path,
    estado: dict[str, Any],
    candidatos: list[ConfigCNN] | None = None,
    semilla: int = SEMILLA,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validación agrupada externa para la CNN con búsqueda interna de hiperparámetros."""
    proyecto = Path(proyecto).resolve()
    salida = Path(salida)
    checkpoints = Path(checkpoints)
    salida.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    rutas = [resolver_ruta_cnn(fila, proyecto) for _, fila in base.iterrows()]
    y_idx = base.calidad_revisada.map(CLASE_A_INDICE).to_numpy()
    grupos = base.grupo_particion.to_numpy()
    candidatos = candidatos or rejilla_cnn_por_defecto()

    from datetime import datetime, timezone

    def registrar(contexto):
        registro = {'fecha_utc': datetime.now(timezone.utc).isoformat(), **contexto}
        if 'epoca' not in registro or registro['epoca'] == 1 or registro['epoca'] % 5 == 0:
            try:
                guardar_json(registro, salida / 'cnn_progreso.json')
            except PermissionError:
                print('El archivo de avance está ocupado; el entrenamiento continúa.', flush=True)
            print('CNN:', registro, flush=True)

    asignacion = np.full(len(base), -1, dtype=int)
    auditoria = []
    for num, (tr, va) in enumerate(particiones, 1):
        if set(grupos[tr]) & set(grupos[va]):
            raise ValueError('Se mezclaron grupos en la evaluación externa de CNN.')
        if (asignacion[va] != -1).any():
            raise ValueError('Una imagen tiene más de un bloque externo.')
        asignacion[va] = num
        for etapa, indices in [('entrenamiento', tr), ('validacion', va)]:
            for indice in indices:
                auditoria.append({'exterior': num, 'interior': 0, 'etapa': etapa,
                                  'id_ejemplar': base.iloc[indice].id_ejemplar, 'grupo': grupos[indice]})
        cv = crear_cv_interno(y_idx[tr], grupos[tr], max_splits=3, semilla=semilla + num * 100)
        for interior, (a, b) in enumerate(cv.split(np.zeros(len(tr)), y_idx[tr], grupos[tr]), 1):
            if set(grupos[tr[a]]) & set(grupos[tr[b]]):
                raise ValueError('Se mezclaron grupos internos de CNN.')
            if set(y_idx[tr[a]]) != set(range(len(CALIDADES))):
                raise ValueError('Faltan calidades en el entrenamiento interno de CNN.')
            for etapa, indices in [('entrenamiento', tr[a]), ('validacion', tr[b])]:
                for indice in indices:
                    auditoria.append({'exterior': num, 'interior': interior, 'etapa': etapa,
                                      'id_ejemplar': base.iloc[indice].id_ejemplar, 'grupo': grupos[indice]})
    if (asignacion < 0).any():
        raise ValueError('Faltan imágenes en la evaluación externa.')
    guardar_tabla(pd.DataFrame(auditoria), salida / 'cnn_particiones.csv')
    guardar_json({'sha_base': estado['sha_base'], 'semilla': semilla,
                  'candidatos': [asdict(c) for c in candidatos], 'pliegues_externos': len(particiones),
                  'seleccion': 'F1 macro interno medio; duración: media redondeada de mejores épocas internas, mínimo 1.',
                  'evaluacion_externa': 'Una sola predicción después del ajuste; no se evalúa el bloque externo por época.'},
                 salida / 'cnn_plan_entrenamiento.json')

    oof = np.full(len(base), '', dtype=object)
    historiales = []
    resumen_pliegues = []
    configuraciones = []
    probabilidades = np.full((len(base), len(CALIDADES)), np.nan)

    def guardar_oof():
        tabla = base[['id_ejemplar', 'grupo_particion', 'fruta', 'calidad_revisada', 'fuente']].copy()
        tabla['pliegue_validacion'], tabla['prediccion'] = asignacion, oof
        for i, clase in enumerate(CALIDADES):
            tabla[f'prob_{clase}'] = probabilidades[:, i]
        guardar_tabla(tabla, salida / 'cnn_predicciones_oof.csv')

    for num_pliegue, (train_idx, val_idx) in enumerate(particiones, 1):
        registrar({'fase': 'seleccion_interna', 'exterior': num_pliegue})
        rutas_tr = [rutas[i] for i in train_idx]
        rutas_va = [rutas[i] for i in val_idx]
        mejor_cfg, tabla_busqueda = seleccionar_hiperparametros_cnn(
            rutas_tr,
            y_idx[train_idx],
            grupos[train_idx],
            candidatos,
            semilla=semilla + num_pliegue * 100,
            progreso=lambda r, n=num_pliegue: registrar({'fase': 'seleccion_interna', 'exterior': n, **r}),
        )
        tabla_busqueda['pliegue_externo'] = num_pliegue
        resumen_pliegues.append(tabla_busqueda)
        configuraciones.append({'exterior': num_pliegue, **asdict(mejor_cfg)})
        guardar_tabla(pd.concat(resumen_pliegues, ignore_index=True), salida / 'cnn_busqueda_interna.csv')
        guardar_tabla(pd.DataFrame(configuraciones), salida / 'cnn_configuraciones_externas.csv')
        registrar({'fase': 'ajuste_externo', 'exterior': num_pliegue, 'epocas': mejor_cfg.max_epocas})

        # Ajuste del pliegue externo con la duración estimada en CV interno, sin usar val_idx para early stopping.
        modelo_pliegue, hist, _ = entrenar_un_ajuste_cnn(
            rutas_tr,
            y_idx[train_idx],
            None,
            None,
            mejor_cfg,
            semilla=semilla + num_pliegue,
            usar_val_para_early_stopping=False,
            progreso=lambda r, n=num_pliegue: registrar({'fase': 'ajuste_externo', 'exterior': n, **r}),
        )
        # Predicción OOF: usamos el último peso o el entrenado sin usar val_idx para seleccionar arquitectura.
        preds_va, probs_va = predecir_rutas_cnn(modelo_pliegue, rutas_va, lado_entrada=mejor_cfg.lado_entrada)
        oof[val_idx] = preds_va
        probabilidades[val_idx] = probs_va
        guardar_oof()
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in modelo_pliegue.state_dict().items()},
                    'config': asdict(mejor_cfg), 'calidades': CALIDADES, 'sha_base': estado['sha_base'],
                    'ids_train': base.iloc[train_idx].id_ejemplar.tolist(),
                    'ids_validacion': base.iloc[val_idx].id_ejemplar.tolist()},
                   checkpoints / f'cnn_pliegue_{num_pliegue}.pt')
        for h in hist:
            historiales.append({'pliegue': num_pliegue, **h})
        guardar_tabla(pd.DataFrame(historiales), salida / 'cnn_historial_epocas.csv')
        registrar({'fase': 'bloque_terminado', 'exterior': num_pliegue})

    _graficar_curvas_cnn(pd.DataFrame(historiales), salida / 'curvas_entrenamiento_cnn.png')

    # Selección y modelo final entrenado sobre toda la base de desarrollo para despliegue.
    registrar({'fase': 'seleccion_global'})
    mejor_global, tabla_global = seleccionar_hiperparametros_cnn(
        rutas, y_idx, grupos, candidatos, semilla=semilla,
        progreso=lambda r: registrar({'fase': 'seleccion_global', **r}))
    registrar({'fase': 'ajuste_final', 'epocas': mejor_global.max_epocas})
    modelo_final, hist_final, _ = entrenar_un_ajuste_cnn(
        rutas, y_idx, None, None, mejor_global, semilla=semilla,
        progreso=lambda r: registrar({'fase': 'ajuste_final', **r})
    )

    ruta_ckpt = checkpoints / 'cnn_calidad.pt'
    torch.save({
        'state_dict': {k: v.detach().cpu() for k, v in modelo_final.state_dict().items()},
        'config': asdict(mejor_global),
        'calidades': CALIDADES,
        'frutas': sorted(base.fruta.unique().tolist()),
        'media_norm': MEDIA_NORM,
        'desv_norm': DESV_NORM,
        'sha_base': estado['sha_base'],
        'alcance': 'Prototipo CNN simple entrenada desde cero sobre recortes 224x224.',
    }, ruta_ckpt)

    pd.concat(resumen_pliegues, ignore_index=True).to_csv(salida / 'cnn_busqueda_interna.csv', index=False)
    tabla_global.to_csv(salida / 'cnn_busqueda_global.csv', index=False)
    pd.DataFrame(historiales).to_csv(salida / 'cnn_historial_epocas.csv', index=False)
    guardar_tabla(pd.DataFrame(hist_final), salida / 'cnn_historial_final.csv')
    if (oof == '').any() or not np.isfinite(probabilidades).all():
        raise ValueError('La CNN no terminó las predicciones externas.')
    registrar({'fase': 'completado', 'imagenes_oof': len(oof)})

    return oof, {
        'mejor_config_global': asdict(mejor_global),
        'checkpoint': str(ruta_ckpt),
        'epocas_finales': len(hist_final),
        'configuraciones_externas': configuraciones,
        'dispositivo': str(next(modelo_final.parameters()).device),
        'hilos_cpu': torch.get_num_threads(),
    }


def _graficar_curvas_cnn(historial: pd.DataFrame, ruta: Path) -> None:
    if historial.empty:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for pliegue, sub in historial.groupby('pliegue'):
        axes[0].plot(sub['epoca'], sub['loss_train'], label=f'P{pliegue} train', linestyle='-')
        if 'loss_val' in sub:
            axes[0].plot(sub['epoca'], sub['loss_val'], label=f'P{pliegue} val', linestyle='--')
        axes[1].plot(sub['epoca'], sub['f1_train'], label=f'P{pliegue} train', linestyle='-')
        if 'f1_val' in sub:
            axes[1].plot(sub['epoca'], sub['f1_val'], label=f'P{pliegue} val', linestyle='--')
    axes[0].set(title='Pérdida por época (CNN)', xlabel='Época', ylabel='CrossEntropy ponderada')
    axes[1].set(title='F1 macro de entrenamiento por época (CNN)', xlabel='Época', ylabel='F1 macro', ylim=(0, 1))
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)
    guardar_grafica(fig, ruta)
    plt.close(fig)


# Mejora del mismo flujo: CNN preentrenada fija y clasificador de calidad entrenable.
MEDIA_IMAGENET = (.485, .456, .406)
DESV_IMAGENET = (.229, .224, .225)
VISTAS_TRANSFERENCIA = ['original', 'espejo', 'giro_expandido_menos_10', 'giro_expandido_mas_10', 'brillo_092']


@dataclass(frozen=True)
class ConfigTransferencia:
    usar_aumentos: bool = True
    cabezal: str = 'lineal'
    potencia_peso: float = 0.
    lr: float = .003
    weight_decay: float = .01
    dropout: float = .2
    max_epocas: int = 120
    min_epocas: int = 12
    paciencia: int = 15


class CNNTransferencia(nn.Module):
    """MobileNetV3 Small fija, con un nuevo cabezal para las tres calidades."""

    def __init__(self, config: ConfigTransferencia, pesos_preentrenados: bool = False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        red = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pesos_preentrenados else None)
        self.extractor = red.features
        self.pool = red.avgpool
        self.config = config
        for parametro in self.extractor.parameters():
            parametro.requires_grad_(False)
        self.cabezal = crear_cabezal(config)
        self.extractor.eval()

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()  # Nunca actualizar BatchNorm con las imágenes de evaluación.
        return self

    def extraer(self, x):
        return self.pool(self.extractor(x)).flatten(1)

    def forward(self, x):
        return self.cabezal(self.extraer(x))


def crear_cabezal(config):
    pasos = [nn.LayerNorm(576, elementwise_affine=False)]
    if config.cabezal == 'mlp':
        pasos += [nn.Linear(576, 64), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(64, 3)]
    elif config.cabezal == 'lineal':
        pasos += [nn.Dropout(config.dropout), nn.Linear(576, 3)]
    else:
        raise ValueError('Cabezal desconocido.')
    return nn.Sequential(*pasos)


def candidatos_transferencia(max_epocas=120):
    return [ConfigTransferencia(cabezal=c, potencia_peso=p, usar_aumentos=a, max_epocas=max_epocas,
                                min_epocas=min(12, max_epocas))
            for c in ['lineal', 'mlp'] for p in [0., .5, 1.] for a in [False, True]]


def transformacion_transferencia():
    # La base ya conserva el encuadre completo con proporciones fijas; no aplicar CenterCrop.
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(MEDIA_IMAGENET, DESV_IMAGENET)])


def predecir_transferencia(modelo, rutas, batch_size=8):
    modelo.eval()
    loader = DataLoader(DatasetFrutasCNN(rutas, None, transformacion_transferencia()),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        probs = torch.cat([modelo(x).softmax(1) for x in loader]).numpy()
    return np.array([CALIDADES[i] for i in probs.argmax(1)]), probs


def extraer_vistas_fijas(modelo, base, estado, salida):
    """Cache de cálculo sin etiquetas: el extractor y sus estadísticas están fijos."""
    from torchvision.transforms import functional as F
    firma = {'sha_base': estado['sha_base'], 'pesos': 'MobileNet_V3_Small_Weights.IMAGENET1K_V1',
             'vistas': VISTAS_TRANSFERENCIA, 'normalizacion': [MEDIA_IMAGENET, DESV_IMAGENET],
             'tamano': 224, 'preparacion': estado.get('entrada_cnn', 'recorte_mascara_historico'),
             'torchvision': __import__('torchvision').__version__}
    ruta = salida / 'cnn_caracteristicas.pt'
    if ruta.is_file():
        cache = torch.load(ruta, map_location='cpu', weights_only=True)
        if cache['firma'] == firma and cache['ids'] == base.id_ejemplar.tolist():
            return cache['vectores'], firma
    modelo.eval()
    normalizar = transformacion_transferencia()
    vectores = []
    for indice, fila in enumerate(base.itertuples(), 1):
        imagen = _imagen_rgb_en_memoria(fila.ruta_cnn, Path(fila.ruta_cnn).stat().st_mtime_ns).copy()
        from src.data.entrada_cnn import preparar_imagen_completa
        giros = [Image.fromarray(preparar_imagen_completa(np.asarray(F.rotate(imagen, angulo, expand=True, fill=128)))) for angulo in [-10, 10]]
        vistas = [imagen, F.hflip(imagen), *giros, F.adjust_brightness(imagen, .92)]
        x = torch.stack([normalizar(v) for v in vistas])
        with torch.inference_mode():
            vectores.append(modelo.extraer(x).clone())
        if indice % 50 == 0:
            print(f'CNN: rasgos visuales fijos {indice}/{len(base)}.', flush=True)
    z = torch.stack(vectores).detach().clone()
    torch.save({'firma': firma, 'ids': base.id_ejemplar.tolist(), 'vectores': z}, ruta)
    return z, firma


def entrenar_cabezal(z_train, y_train, config, semilla, z_val=None, y_val=None):
    """Entrenar solo con imágenes de entrenamiento; validar siempre la vista original."""
    fijar_semillas(semilla)
    cabeza = crear_cabezal(config)
    y = torch.tensor(np.asarray(y_train).copy(), dtype=torch.long)
    conteos = torch.bincount(y, minlength=3).float().clamp_min(1)
    pesos = (len(y) / (3 * conteos)).pow(config.potencia_peso)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    optimizador = torch.optim.AdamW(cabeza.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mejor, perdida_mejor, epoca_mejor, sin_mejora = -1., float('inf'), 0, 0
    mejor_estado, historial = None, []
    n = len(y)
    for epoca in range(1, config.max_epocas + 1):
        cabeza.train()
        # Un aumento por imagen, sin multiplicar su peso ni usar vistas de validación.
        vistas = torch.randint(z_train.shape[1] if config.usar_aumentos else 1, (n,))
        x = z_train[torch.arange(n), vistas]
        optimizador.zero_grad()
        logits = cabeza(x)
        perdida = criterio(logits, y)
        perdida.backward()
        optimizador.step()
        registro = {'epoca': epoca, 'loss_train': float(perdida.detach()),
                    'f1_train': float(f1_score(y.numpy(), logits.detach().argmax(1).numpy(),
                                              labels=[0, 1, 2], average='macro', zero_division=0))}
        if z_val is not None:
            cabeza.eval()
            with torch.no_grad():
                salida = cabeza(z_val[:, 0])
                loss_val = float(nn.functional.cross_entropy(salida, torch.tensor(np.asarray(y_val).copy(), dtype=torch.long)))
            f1 = float(f1_score(y_val, salida.argmax(1).numpy(), labels=[0, 1, 2], average='macro', zero_division=0))
            registro.update(loss_val=loss_val, f1_val=f1)
            if epoca >= config.min_epocas:
                mejora = f1 > mejor + 1e-12 or (abs(f1 - mejor) <= 1e-12 and loss_val < perdida_mejor - 1e-6)
                if mejora:
                    mejor, perdida_mejor, epoca_mejor, sin_mejora = f1, loss_val, epoca, 0
                    mejor_estado = {k: v.detach().clone() for k, v in cabeza.state_dict().items()}
                else:
                    sin_mejora += 1
        else:
            epoca_mejor = epoca
        historial.append(registro)
        if z_val is not None and sin_mejora >= config.paciencia:
            break
    if mejor_estado is not None:
        cabeza.load_state_dict(mejor_estado)
    return cabeza, historial, {'mejor_epoca': epoca_mejor, 'mejor_f1_val': mejor}


def evaluar_transferencia_anidada(base, particiones, proyecto, salida, checkpoints, estado,
                                  candidatos=None, semilla=42):
    """Mejora de CNN dentro del mismo experimento y la misma base."""
    from datetime import datetime, timezone
    proyecto, salida, checkpoints = Path(proyecto), Path(salida), Path(checkpoints)
    salida.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(proyecto / 'data/external/torch'))
    candidatos = candidatos or candidatos_transferencia()
    red = CNNTransferencia(candidatos[0], pesos_preentrenados=True).eval()
    z, firma = extraer_vistas_fijas(red, base, estado, salida)
    y = base.calidad_revisada.map(CLASE_A_INDICE).to_numpy()
    g = base.grupo_particion.to_numpy()
    asignacion = np.full(len(base), -1)
    for k, (tr, va) in enumerate(particiones, 1):
        if set(g[tr]) & set(g[va]):
            raise ValueError('Grupos mezclados en la evaluación externa.')
        asignacion[va] = k
    if (asignacion < 0).any():
        raise ValueError('Faltan imágenes de evaluación.')
    guardar_json({'sha_base': estado['sha_base'], 'semilla': semilla, 'arquitectura': 'mobilenet_v3_small_congelada',
                  'candidatos': [asdict(c) for c in candidatos], 'extractor': firma,
                  'criterio': 'F1 macro interno; desempate por menor pérdida interna no ponderada.',
                  'seleccion_epocas': 'Media redondeada de mejores épocas internas, con mínimo de calentamiento.',
                  'evaluacion': 'Fotografía completa. Aumentos opcionales elegidos por validación interna; siempre sin recorte.'},
                 salida / 'cnn_plan_entrenamiento.json')
    registros, auditoria, configs, historiales = [], [], [], []

    def progreso(**datos):
        try:
            guardar_json({'fecha_utc': datetime.now(timezone.utc).isoformat(), **datos}, salida / 'cnn_progreso.json')
        except PermissionError:
            pass
        print('CNN transferencia:', datos, flush=True)

    def seleccionar(indices, exterior, seed):
        cv = crear_cv_interno(y[indices], g[indices], max_splits=3, semilla=seed)
        internos = list(cv.split(np.zeros(len(indices)), y[indices], g[indices]))
        for interior, (a, b) in enumerate(internos, 1):
            if set(g[indices[a]]) & set(g[indices[b]]):
                raise ValueError('Grupos mezclados internamente.')
            if set(y[indices[a]]) != {0, 1, 2}:
                raise ValueError('Faltan clases en el entrenamiento interno.')
            for etapa, sub in [('entrenamiento', indices[a]), ('validacion', indices[b])]:
                for i in sub:
                    auditoria.append({'exterior': exterior, 'interior': interior, 'etapa': etapa,
                                      'id_ejemplar': base.iloc[i].id_ejemplar, 'grupo': g[i]})
        resultados = []
        for numero, cfg in enumerate(candidatos, 1):
            progreso(fase='seleccion_interna' if exterior else 'seleccion_global', exterior=exterior, candidato=numero)
            f1s, epocas = [], []
            for interior, (a, b) in enumerate(internos, 1):
                _, _, info = entrenar_cabezal(z[indices[a]], y[indices[a]], cfg, seed + interior,
                                               z[indices[b]], y[indices[b]])
                f1s.append(info['mejor_f1_val'])
                epocas.append(info['mejor_epoca'])
            resultados.append({**asdict(cfg), 'candidato': numero, 'f1_macro_interno_medio': float(np.mean(f1s)),
                               'f1_macro_interno_std': float(np.std(f1s)), 'epoca_media': int(round(np.mean(epocas)))})
        tabla = pd.DataFrame(resultados).sort_values('f1_macro_interno_medio', ascending=False, kind='stable')
        fila = tabla.iloc[0]
        cfg = candidatos[int(fila.candidato) - 1]
        from dataclasses import replace
        elegida = replace(cfg, max_epocas=max(cfg.min_epocas, min(cfg.max_epocas, int(fila.epoca_media))))
        return elegida, tabla

    oof = np.full(len(base), '', dtype=object)
    probabilidades = np.full((len(base), 3), np.nan)
    for exterior, (tr, va) in enumerate(particiones, 1):
        for etapa, indices in [('entrenamiento', tr), ('validacion', va)]:
            for i in indices:
                auditoria.append({'exterior': exterior, 'interior': 0, 'etapa': etapa,
                                  'id_ejemplar': base.iloc[i].id_ejemplar, 'grupo': g[i]})
        cfg, tabla = seleccionar(tr, exterior, semilla + exterior * 100)
        tabla['pliegue_externo'] = exterior
        registros.append(tabla)
        configs.append({'exterior': exterior, **asdict(cfg)})
        cabeza, hist, _ = entrenar_cabezal(z[tr], y[tr], cfg, semilla + exterior)
        red.cabezal = cabeza
        red.config = cfg
        red.eval()
        etiquetas, prob = predecir_transferencia(red, [Path(p) for p in base.iloc[va].ruta_cnn])
        probabilidades[va] = prob
        oof[va] = etiquetas
        torch.save({'arquitectura': 'mobilenet_v3_small_congelada', 'preparacion_cnn': estado.get('entrada_cnn'), 'state_dict': red.state_dict(),
                    'config': asdict(cfg), 'calidades': CALIDADES, 'sha_base': estado['sha_base'],
                    'ids_train': base.iloc[tr].id_ejemplar.tolist(), 'ids_validacion': base.iloc[va].id_ejemplar.tolist()},
                   checkpoints / f'cnn_pliegue_{exterior}.pt')
        historiales.extend([{'pliegue': exterior, **r} for r in hist])
        guardar_tabla(pd.concat(registros, ignore_index=True), salida / 'cnn_busqueda_interna.csv')
        guardar_tabla(pd.DataFrame(configs), salida / 'cnn_configuraciones_externas.csv')
        progreso(fase='bloque_terminado', exterior=exterior)
    if (oof == '').any():
        raise ValueError('Faltan predicciones externas.')
    tabla_oof = base[['id_ejemplar', 'grupo_particion', 'fruta', 'calidad_revisada', 'fuente']].copy()
    tabla_oof['pliegue_validacion'], tabla_oof['prediccion'] = asignacion.astype(int), oof
    for i, c in enumerate(CALIDADES):
        tabla_oof[f'prob_{c}'] = probabilidades[:, i]
    guardar_tabla(tabla_oof, salida / 'cnn_predicciones_oof.csv')
    cfg, tabla_global = seleccionar(np.arange(len(base)), 0, semilla)
    cabeza, hist_final, _ = entrenar_cabezal(z, y, cfg, semilla)
    red.cabezal, red.config = cabeza, cfg
    torch.save({'arquitectura': 'mobilenet_v3_small_congelada', 'preparacion_cnn': estado.get('entrada_cnn'), 'state_dict': red.state_dict(), 'config': asdict(cfg),
                'calidades': CALIDADES, 'frutas': sorted(base.fruta.unique().tolist()), 'sha_base': estado['sha_base'],
                'media_norm': MEDIA_IMAGENET, 'desv_norm': DESV_IMAGENET,
                'pesos_iniciales': firma['pesos'], 'alcance': 'CNN de calidad con extractor preentrenado fijo.'},
               checkpoints / 'cnn_calidad.pt')
    guardar_tabla(tabla_global, salida / 'cnn_busqueda_global.csv')
    guardar_tabla(pd.DataFrame(historiales), salida / 'cnn_historial_epocas.csv')
    guardar_tabla(pd.DataFrame(hist_final), salida / 'cnn_historial_final.csv')
    guardar_tabla(pd.DataFrame(auditoria), salida / 'cnn_particiones.csv')
    _graficar_curvas_cnn(pd.DataFrame(historiales), salida / 'curvas_entrenamiento_cnn.png')
    progreso(fase='completado', imagenes_oof=len(base))
    return oof, {'arquitectura': 'mobilenet_v3_small_congelada', 'mejor_config_global': asdict(cfg),
                 'checkpoint': str(checkpoints / 'cnn_calidad.pt'), 'epocas_finales': len(hist_final),
                 'configuraciones_externas': configs, 'dispositivo': 'cpu', 'hilos_cpu': torch.get_num_threads(),
                 'extractor_fijo': firma, 'parametros_entrenables': sum(p.numel() for p in cabeza.parameters())}
