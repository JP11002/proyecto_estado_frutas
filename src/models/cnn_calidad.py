"""CNN simple en PyTorch entrenada desde cero sobre recortes RGB 224x224.

Usa la salida de `preparar_cnn` (aspecto preservado con relleno neutro 128),
ponderación inversa de clases por entrenamiento, aumento ligero exclusivo de
entrenamiento y validación agrupada sin fuga de frutos.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
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

from src.models.evaluacion import (
    CALIDADES,
    SEMILLA,
    crear_cv_interno,
    guardar_grafica,
)

CLASE_A_INDICE = {c: i for i, c in enumerate(CALIDADES)}
INDICE_A_CLASE = {i: c for i, c in enumerate(CALIDADES)}
MEDIA_NORM = (0.5, 0.5, 0.5)
DESV_NORM = (0.5, 0.5, 0.5)


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
    """Resolver la ruta de entrada_cnn.png tolerando cambios de máquina/directorio."""
    ruta = Path(str(fila['ruta_cnn']))
    if ruta.is_file():
        return ruta
    candidatos = [
        proyecto / 'data/processed/drive_20261124/preparacion' / str(fila.get('id_origen', '')) / str(fila.get('metodo', '')) / 'entrada_cnn.png',
        proyecto / 'data/processed/preparacion_imagenes' / str(fila['id_ejemplar']),
    ]
    for cand in candidatos:
        if cand.is_file():
            return cand
        if cand.is_dir():
            encontrado = next(cand.rglob('entrada_cnn.png'), None)
            if encontrado is not None:
                return encontrado
    # Último recurso: buscar la subruta a partir de data/processed si existiera en otro árbol.
    partes = ruta.parts
    if 'data' in partes:
        rel = Path(*partes[partes.index('data'):])
        en_proyecto = proyecto / rel
        if en_proyecto.is_file():
            return en_proyecto
    raise FileNotFoundError(f'No se encontró entrada_cnn.png para {fila["id_ejemplar"]} ({ruta}).')


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
        with Image.open(self.rutas[idx]) as im:
            rgb = im.convert('RGB')
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
            perdida_total += float(perdida.item()) * len(y)
            total += len(y)
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
            perdida_sum += float(perdida.item()) * len(y)
            n_muestras += len(y)
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
) -> tuple[ConfigCNN, pd.DataFrame]:
    """Buscar mejores hiperparámetros para la CNN mediante validación interna agrupada."""
    cv_interno = crear_cv_interno(y_idx, grupos, max_splits=3, semilla=semilla)
    splits_internos = list(cv_interno.split(np.zeros(len(y_idx)), y_idx, grupos))
    registros = []

    for idx_cfg, cfg in enumerate(candidatos):
        f1_pliegues = []
        epocas_opt = []
        for idx_sub, (tr_sub, va_sub) in enumerate(splits_internos):
            rutas_tr = [rutas[i] for i in tr_sub]
            rutas_va = [rutas[i] for i in va_sub]
            _, _, info = entrenar_un_ajuste_cnn(
                rutas_tr,
                y_idx[tr_sub],
                rutas_va,
                y_idx[va_sub],
                cfg,
                semilla=semilla + idx_cfg * 10 + idx_sub,
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
        max_epocas=max(5, int(mejor_fila['epoca_media'])),
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

    oof = np.full(len(base), '', dtype=object)
    historiales = []
    resumen_pliegues = []

    for num_pliegue, (train_idx, val_idx) in enumerate(particiones, 1):
        rutas_tr = [rutas[i] for i in train_idx]
        rutas_va = [rutas[i] for i in val_idx]
        mejor_cfg, tabla_busqueda = seleccionar_hiperparametros_cnn(
            rutas_tr,
            y_idx[train_idx],
            grupos[train_idx],
            candidatos,
            semilla=semilla + num_pliegue * 100,
        )
        tabla_busqueda['pliegue_externo'] = num_pliegue
        resumen_pliegues.append(tabla_busqueda)

        # Ajuste del pliegue externo con la duración estimada en CV interno, sin usar val_idx para early stopping.
        modelo_pliegue, hist, _ = entrenar_un_ajuste_cnn(
            rutas_tr,
            y_idx[train_idx],
            rutas_va,
            y_idx[val_idx],
            mejor_cfg,
            semilla=semilla + num_pliegue,
            usar_val_para_early_stopping=False,
        )
        # Predicción OOF: usamos el último peso o el entrenado sin usar val_idx para seleccionar arquitectura.
        preds_va, _ = predecir_rutas_cnn(modelo_pliegue, rutas_va, lado_entrada=mejor_cfg.lado_entrada)
        oof[val_idx] = preds_va
        for h in hist:
            historiales.append({'pliegue': num_pliegue, **h})

    _graficar_curvas_cnn(pd.DataFrame(historiales), salida / 'curvas_entrenamiento_cnn.png')

    # Selección y modelo final entrenado sobre toda la base de desarrollo para despliegue.
    mejor_global, tabla_global = seleccionar_hiperparametros_cnn(rutas, y_idx, grupos, candidatos, semilla=semilla)
    modelo_final, hist_final, _ = entrenar_un_ajuste_cnn(
        rutas, y_idx, None, None, mejor_global, semilla=semilla
    )

    ruta_ckpt = checkpoints / 'cnn_calidad.pt'
    torch.save({
        'state_dict': modelo_final.state_dict(),
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

    return oof, {
        'mejor_config_global': asdict(mejor_global),
        'checkpoint': str(ruta_ckpt),
        'epocas_finales': len(hist_final),
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
    axes[1].set(title='F1 macro por época (CNN)', xlabel='Época', ylabel='F1 macro', ylim=(0, 1))
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)
    guardar_grafica(fig, ruta)
    plt.close(fig)
