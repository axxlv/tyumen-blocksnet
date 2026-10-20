#!/usr/bin/env python
"""Собрать финальный HTML dashboard по Тюмени только из refined-результатов.

Скрипт:
- читает только готовые refined-файлы;
- не обращается к OSM, Overpass, API или сети;
- не пересчитывает матрицу доступности и расчётные метрики;
- не использует original-слои и comparison-таблицы;
- строит одну интерактивную Leaflet-карту, на которой переключаются
  сохранённые refined-показатели, графики и таблицы.

Запуск из корня репозитория:
    .\\.venv-blocksnet\\Scripts\\python.exe .\\examples\\tyumen_dashboard_refined.py
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from datetime import datetime
from pathlib import Path
from textwrap import fill

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_ROOT = EXAMPLES_DIR / "data_tyumen"
REFINED_DIR = DATA_ROOT / "refined"
OUTPUT = DATA_ROOT / "dashboard_refined.html"
PREVIOUS_OUTPUT = DATA_ROOT / "dashboard_refined_previous.html"

REQUIRED = [
    REFINED_DIR / "results_table.csv",
    REFINED_DIR / "results_summary.csv",
    REFINED_DIR / "provision_schools.csv",
    REFINED_DIR / "provision_links.csv",
    REFINED_DIR / "morphotypes_spacematrix.csv",
    REFINED_DIR / "morphotypes_spacematrix_clusters.csv",
    REFINED_DIR / "morphotypes_strelka.csv",
    REFINED_DIR / "area_accessibility.csv",
    REFINED_DIR / "blocks_land_use.geojson",
    REFINED_DIR / "buildings_blocks.geojson",
    REFINED_DIR / "schools.geojson",
    REFINED_DIR / "schools_blocks.geojson",
    REFINED_DIR / "water.geojson",
    REFINED_DIR / "boundaries.geojson",
    REFINED_DIR / "quality_report.json",
    REFINED_DIR / "blocks_quality.json",
    REFINED_DIR / "blocks_quality.csv",
    REFINED_DIR / "matrix_meta.json",
    REFINED_DIR / "osm_source.json",
    REFINED_DIR / "split_meta.json",
]


def fail(message: str) -> None:
    raise SystemExit(message)


missing = [str(path) for path in REQUIRED if not path.exists()]
if missing:
    fail("Не хватает файлов refined:\n" + "\n".join(missing))


def read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    return df


def read_blocks(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if "id" in gdf.columns:
        gdf = gdf.set_index("id")
    gdf.index = gdf.index.astype(int)
    return gdf


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits: int = 2) -> str:
    """Аккуратно отформатировать число, NaN, бесконечность или список.

    Разряды разделяются неразрывным пробелом, пропуски показываются как «—»,
    а бесконечные значения — отдельным знаком, чтобы их нельзя было спутать
    со средним значением.
    """
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(item, digits) for item in value)
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(number):
        return "∞" if number > 0 else "−∞"
    space = "\u00a0"
    if abs(number) >= 1000:
        if digits == 0:
            return f"{number:,.0f}".replace(",", space)
        return f"{number:,.{digits}f}".replace(",", space)
    return f"{number:,.{digits}f}".replace(",", space)


def plural(count: int, one: str, few: str, many: str) -> str:
    """Согласовать существительное с числом: 1 квартал, 2 квартала, 5 кварталов."""
    count = abs(int(count))
    if count % 10 == 1 and count % 100 != 11:
        return one
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return few
    return many


def table_html(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_html(
        classes="data-table",
        border=0,
        escape=True,
        index=index,
        justify="left",
    )


def image_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def chart_image(title: str, description: str, fig) -> str:
    uri = image_data_uri(fig)
    return (
        '<figure class="chart-card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<img src="{uri}" alt="{html.escape(title)}">'
        f"<figcaption>{html.escape(description)}</figcaption>"
        "</figure>"
    )


def js_json(value) -> str:
    """JSON для вставки в <script>. Закрывающие теги и U+2028 экранируются."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


# ---------------------------------------------------------------------------
# Только сохранённые refined-данные
# ---------------------------------------------------------------------------
ref_results = read_results(REFINED_DIR / "results_table.csv")
ref_blocks = read_blocks(REFINED_DIR / "blocks_land_use.geojson")
schools = gpd.read_file(REFINED_DIR / "schools.geojson")
water = gpd.read_file(REFINED_DIR / "water.geojson")
boundaries = gpd.read_file(REFINED_DIR / "boundaries.geojson")
buildings_blocks = read_blocks(REFINED_DIR / "buildings_blocks.geojson")
schools_blocks = read_blocks(REFINED_DIR / "schools_blocks.geojson")
spacematrix = pd.read_csv(REFINED_DIR / "morphotypes_spacematrix.csv", index_col=0)
spacematrix.index = spacematrix.index.astype(int)
strelka = pd.read_csv(REFINED_DIR / "morphotypes_strelka.csv", index_col=0)
strelka.index = strelka.index.astype(int)
spacematrix_clusters = pd.read_csv(REFINED_DIR / "morphotypes_spacematrix_clusters.csv")
provision_schools = pd.read_csv(REFINED_DIR / "provision_schools.csv", index_col=0)
provision_schools.index = provision_schools.index.astype(int)
provision_links = pd.read_csv(REFINED_DIR / "provision_links.csv")
area_accessibility = pd.read_csv(REFINED_DIR / "area_accessibility.csv", index_col=0)
area_accessibility.index = area_accessibility.index.astype(int)

quality_report = read_json(REFINED_DIR / "quality_report.json")
blocks_quality = read_json(REFINED_DIR / "blocks_quality.json")
matrix_meta = read_json(REFINED_DIR / "matrix_meta.json")
osm_source = read_json(REFINED_DIR / "osm_source.json")
split_meta = read_json(REFINED_DIR / "split_meta.json")
results_summary = pd.read_csv(REFINED_DIR / "results_summary.csv", index_col=0)
blocks_quality_rows = pd.read_csv(REFINED_DIR / "blocks_quality.csv")

# Метрическая проекция нужна только для площади и аккуратной отрисовки.
utm = ref_blocks.estimate_utm_crs()
for layer in (schools, water, boundaries):
    if layer.crs is None:
        layer = layer.set_crs(4326)

# Базовая таблица кварталов: геометрия + сохранённые результаты.
base = ref_blocks[["geometry"]].join(ref_results, how="left")
base["id"] = base.index.astype(int)
base["area_km2"] = pd.to_numeric(base["site_area"], errors="coerce") / 1_000_000
base["area_ha"] = pd.to_numeric(base["site_area"], errors="coerce") / 10_000
base["population_density"] = base["population"] / base["area_km2"]

# Дополнительные сохранённые агрегаты: число зданий и школ.
if "count" in buildings_blocks.columns:
    base["building_count"] = buildings_blocks["count"].reindex(base.index)
else:
    base["building_count"] = np.nan
if "count" in schools_blocks.columns:
    base["school_count"] = schools_blocks["count"].reindex(base.index)
elif "n_objects" in schools_blocks.columns:
    base["school_count"] = schools_blocks["n_objects"].reindex(base.index)
else:
    base["school_count"] = np.nan

# Часть показателей обеспеченности сохранена отдельно от results_table.csv.
for column in [
    "demand",
    "demand_left",
    "demand_within",
    "demand_without",
    "capacity_left",
    "capacity_within",
    "capacity_without",
]:
    if column not in base.columns and column in provision_schools.columns:
        base[column] = provision_schools[column].reindex(base.index)

# Кластер Spacematrix сохранён в morphotypes_spacematrix.csv.
if "cluster" in spacematrix.columns:
    cluster = pd.to_numeric(spacematrix["cluster"], errors="coerce").reindex(base.index)
    base["spacematrix_cluster_label"] = cluster.map(
        lambda value: None if pd.isna(value) else f"Кластер {int(value)}"
    )
else:
    base["spacematrix_cluster_label"] = None

# Площади в гектарах для читаемых карт и подсказок.
base["footprint_area_ha"] = base["footprint_area"] / 10_000
base["build_floor_area_ha"] = base["build_floor_area"] / 10_000
base["living_area_ha"] = base["living_area"] / 10_000
base["non_living_area_ha"] = base["non_living_area"] / 10_000
base["site_area_ha"] = base["site_area"] / 10_000

for column in base.select_dtypes(include=[np.number]).columns:
    base[column] = pd.to_numeric(base[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

# Строковые пропуски должны стать null в GeoJSON, а не строкой "nan".
for column in [
    "land_use",
    "morphotype_spacematrix",
    "morphotype_strelka",
    "spacematrix_cluster_label",
]:
    base[column] = base[column].where(base[column].notna(), None)

# ---------------------------------------------------------------------------
# Подписи категорий
# ---------------------------------------------------------------------------
LAND_USE_LABELS = {
    "residential": "жилая",
    "business": "деловая",
    "recreation": "рекреационная",
    "industrial": "промышленная",
    "transport": "транспортная",
    "special": "специальная",
    "agriculture": "сельскохозяйственная",
}
SPACEMATRIX_LABELS = {
    "low-rise low-density non-residential": "малоэтажная низкоплотная нежилая",
    "low-rise low-density residential": "малоэтажная низкоплотная жилая",
    "low-rise low-density mixed-use": "малоэтажная низкоплотная смешанная",
    "mid-rise non-residential": "среднеэтажная нежилая",
    "mid-rise residential": "среднеэтажная жилая",
    "high-mid-rise residential": "повышенной этажности жилая",
    "high-rise high-density residential": "высотная высокоплотная жилая",
    "high-mid-rise high-density residential": "повышенной этажности высокоплотная жилая",
    "high-mid-rise high-density non-residential": "повышенной этажности высокоплотная нежилая",
}
# Русские названия показателей и единицы измерения для таблиц.
METRIC_TITLES = {
    "population": ("Население квартала", "чел."),
    "population_density": ("Плотность населения", "чел./км²"),
    "building_count": ("Число зданий", "шт."),
    "school_count": ("Число школ", "шт."),
    "capacity": ("Ёмкость школ", "мест"),
    "provision_strong": ("Обеспеченность школами (в пределах 15 минут)", "доля 0–1"),
    "provision_weak": ("Обеспеченность школами (с учётом дальних школ)", "доля 0–1"),
    "demand": ("Нормативная потребность в местах", "мест"),
    "demand_within": ("Покрыто в пределах 15 минут", "мест"),
    "demand_without": ("Покрыто только дальними школами", "мест"),
    "demand_left": ("Непокрытая потребность (дефицит)", "мест"),
    "capacity_within": ("Ёмкость, использованная в пределах 15 минут", "мест"),
    "capacity_without": ("Ёмкость, использованная вне норматива", "мест"),
    "capacity_left": ("Остаток школьных мест", "мест"),
    "area_accessibility": ("Пешеходная доступность территории", "мин"),
    "fsi": ("FSI — интенсивность застройки", "м²/м²"),
    "gsi": ("GSI — доля территории под зданиями", "доля 0–1"),
    "mxi": ("MXI — доля жилой площади", "доля 0–1"),
    "l": ("L — средняя этажность", "этажи"),
    "osr": ("OSR — обеспеченность открытыми пространствами", "м²/м²"),
    "share_living": ("Жилая площадь на единицу пятна застройки", "м²/м²"),
    "share_non_living": ("Нежилая площадь на единицу пятна застройки", "м²/м²"),
    "site_area": ("Площадь квартала", "м²"),
    "footprint_area": ("Площадь пятен застройки", "м²"),
    "build_floor_area": ("Суммарная поэтажная площадь", "м²"),
    "living_area": ("Жилая площадь", "м²"),
    "non_living_area": ("Нежилая площадь", "м²"),
    "residential": ("Доля жилой функции", "доля 0–1"),
    "business": ("Доля деловой функции", "доля 0–1"),
    "recreation": ("Доля рекреационной функции", "доля 0–1"),
    "industrial": ("Доля промышленной функции", "доля 0–1"),
    "transport": ("Доля транспортной функции", "доля 0–1"),
    "special": ("Доля специальной функции", "доля 0–1"),
    "agriculture": ("Доля сельскохозяйственной функции", "доля 0–1"),
    "share": ("Доля преобладающей функции", "доля 0–1"),
    "land_use": ("Преобладающее землепользование", "категория"),
    "morphotype_spacematrix": ("Морфотип Spacematrix", "категория"),
    "morphotype_strelka": ("Морфотип Strelka", "категория"),
    "spacematrix_cluster_label": ("Кластер Spacematrix", "категория"),
}

SUMMARY_COLUMNS = {
    "count": "Кварталов",
    "mean": "Среднее",
    "std": "Стандартное отклонение",
    "min": "Минимум",
    "25%": "25-й перцентиль",
    "50%": "Медиана",
    "75%": "75-й перцентиль",
    "max": "Максимум",
}

STRELKA_LABELS = {
    "low-rise non-residential": "малоэтажная нежилая",
    "low-rise model": "малоэтажная модель",
    "individual residential": "индивидуальная жилая",
    "high-rise modern microdistrict": "высотный современный микрорайон",
    "mid-rise": "среднеэтажная",
    "mid-rise non-residential": "среднеэтажная нежилая",
    "central model": "центральная модель",
    "high-rise non-residential": "высотная нежилая",
    "high-rise soviet microdistrict": "высотный советский микрорайон",
    "mid-rise block": "среднеэтажный квартал",
    "mid-rise microdistrict": "среднеэтажный микрорайон",
}

# ---------------------------------------------------------------------------
# Конфигурация переключателя и легенд
# ---------------------------------------------------------------------------
BLUE = ["#9fc5f2", "#60a5fa", "#2563eb", "#1e40af", "#172554"]
ORANGE = ["#fdba74", "#fb923c", "#f97316", "#c2410c", "#7c2d12"]
PURPLE = ["#ddd6fe", "#c4b5fd", "#7c3aed", "#5b21b6", "#3b0764"]
BROWN = ["#e3cdb8", "#d6b79c", "#b8875f", "#8b5e3c", "#613f27"]
MAGENTA = ["#fbcfe8", "#f9a8d4", "#db2777", "#9d174d", "#500724"]
SLATE = ["#cbd5e1", "#94a3b8", "#64748b", "#334155", "#0f172a"]
CATEGORY_COLORS = [
    "#3b5b92",
    "#d97706",
    "#b91c1c",
    "#7e57c2",
    "#8d6e63",
    "#c9a227",
    "#4b5563",
    "#be185d",
    "#0369a1",
    "#9a3412",
    "#6d28d9",
    "#a0aec0",
]
MISSING_COLOR = "#ffffff"   # нет данных — чисто белый квартал
ZERO_COLOR = "#f0f0f0"     # настоящий ноль — светло-серый
WATER_COLOR = "#707070"    # вода — тёмно-серая заливка
OUTLINE_COLOR = "#8a8a8a"  # границы кварталов — серые линии

metric_config: dict[str, dict] = {}


def clean_breaks(values) -> list[float]:
    """Упорядоченные уникальные границы классов."""
    out: list[float] = []
    for value in values:
        number = float(value)
        if not np.isfinite(number):
            continue
        if not out or number > out[-1] + 1e-12:
            out.append(number)
    if len(out) < 2:
        low = out[0] if out else 0.0
        out = [low, low + 1.0]
    return out


def short_sentence(text: str, limit: int = 180) -> str:
    """Первое предложение описания — короткая интерпретация для подсказки."""
    sentence = text.strip().split(". ")[0].strip()
    if sentence and not sentence.endswith("."):
        sentence += "."
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence


def palette_colors(palette: list[str], intervals: int) -> list[str]:
    """Равномерно взять цвета шкалы, чтобы крайние оттенки всегда использовались."""
    if intervals <= 0:
        return [palette[0]]
    if intervals == 1:
        return [palette[len(palette) // 2]]
    step = (len(palette) - 1) / (intervals - 1)
    return [
        palette[min(len(palette) - 1, int(round(index * step)))]
        for index in range(intervals)
    ]


def add_numeric(
    key: str,
    column: str,
    label: str,
    unit: str,
    group: str,
    description: str,
    high: str,
    low: str,
    decimals: int = 2,
    palette: list[str] | None = None,
    fixed_breaks: list[float] | None = None,
    zero_separate: bool = False,
    zero_label: str = "0",
) -> None:
    palette = palette or BLUE
    series = pd.to_numeric(base[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = series.dropna()
    zero_count = int((valid == 0).sum())
    missing_count = int(series.isna().sum())
    n = int(valid.notna().sum())
    data_min = float(valid.min()) if n else None
    data_max = float(valid.max()) if n else None

    if n == 0:
        cfg_type = "constant"
        breaks = []
        colors = [palette[0]]
        constant_value = None
    elif valid.nunique() <= 1:
        cfg_type = "constant"
        breaks = []
        colors = [palette[0]]
        constant_value = float(valid.iloc[0])
    else:
        cfg_type = "numeric"
        constant_value = None
        if fixed_breaks is not None:
            breaks = clean_breaks(fixed_breaks)
        elif zero_separate and zero_count:
            positive = valid[valid > 0]
            if len(positive) >= 5:
                breaks = clean_breaks(
                    np.quantile(positive, [0.0, 0.25, 0.50, 0.75, 0.90, 1.0])
                )
            elif len(positive):
                breaks = clean_breaks([positive.min(), positive.max()])
            else:
                breaks = [0.0, 1.0]
        else:
            breaks = clean_breaks(np.quantile(valid, [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]))
        n_intervals = max(1, len(breaks) - 1)
        colors = palette_colors(palette, n_intervals)

    metric_config[key] = {
        "key": column,
        "label": label,
        "unit": unit,
        "group": group,
        "type": cfg_type,
        "description": description,
        "short": short_sentence(description),
        "high": high,
        "low": low,
        "decimals": decimals,
        "breaks": breaks,
        "colors": colors,
        "category_colors": [],
        "categories": [],
        "data_min": data_min,
        "data_max": data_max,
        "n": n,
        "missing_count": missing_count,
        "zero_count": zero_count,
        "zero_separate": bool(zero_separate and zero_count),
        "zero_color": ZERO_COLOR,
        "zero_label": zero_label,
        "missing_color": MISSING_COLOR,
        "constant_value": constant_value,
    }


def add_categorical(
    key: str,
    column: str,
    label: str,
    group: str,
    description: str,
    labels: dict[str, str] | None = None,
) -> None:
    labels = labels or {}
    series = base[column].where(base[column].notna(), None)
    counts = series.dropna().astype(str).value_counts()
    categories = []
    for index, (value, count) in enumerate(counts.items()):
        categories.append(
            {
                "value": str(value),
                "label": labels.get(str(value), str(value)),
                "count": int(count),
                "color": CATEGORY_COLORS[index % len(CATEGORY_COLORS)],
            }
        )
    metric_config[key] = {
        "key": column,
        "label": label,
        "unit": "категория",
        "group": group,
        "type": "categorical",
        "description": description,
        "short": short_sentence(description),
        "high": "Категориальная шкала: цвет обозначает категорию, а не величину.",
        "low": "Для каждой категории в легенде указано число кварталов.",
        "decimals": 0,
        "breaks": [],
        "colors": [],
        "category_colors": [item["color"] for item in categories],
        "categories": categories,
        "data_min": None,
        "data_max": None,
        "n": int(series.notna().sum()),
        "missing_count": int(series.isna().sum()),
        "zero_count": 0,
        "zero_separate": False,
        "zero_color": ZERO_COLOR,
        "zero_label": "0",
        "missing_color": MISSING_COLOR,
        "constant_value": None,
    }


add_numeric(
    "population",
    "population",
    "Население квартала",
    "чел.",
    "Население и застройка",
    "Модельная оценка числа жителей в квартале. Это не официальная статистика: население распределено по жилой площади зданий и относится к модельной оценке.",
    "Больше жителей — выше нагрузка на школы, дворы и местную инфраструктуру.",
    "Меньше жителей — квартал менее населён; ноль означает отсутствие жилой функции.",
    decimals=0,
    palette=BLUE,
    zero_separate=True,
    zero_label="0 — населения нет",
)
add_numeric(
    "population_density",
    "population_density",
    "Плотность населения",
    "чел./км²",
    "Население и застройка",
    "Число жителей квартала, делённое на площадь квартала. Это производная величина от двух сохранённых показателей, а не отдельный расчёт.",
    "Высокая плотность — много жителей на единицу площади.",
    "Низкая плотность — население разрежено или отсутствует.",
    decimals=0,
    palette=BLUE,
    zero_separate=True,
    zero_label="0 — населения нет",
)
add_numeric(
    "building_count",
    "building_count",
    "Число зданий",
    "шт.",
    "Население и застройка",
    "Количество зданий в квартале, агрегированное по зданиям из OpenStreetMap.",
    "Больше зданий — квартал сильнее расчленён застройкой.",
    "Ноль — в квартале нет зданий.",
    decimals=0,
    palette=SLATE,
    zero_separate=True,
    zero_label="0 — зданий нет",
)
add_numeric(
    "capacity",
    "capacity",
    "Ёмкость школ в квартале",
    "мест",
    "Школы и обеспеченность",
    "Суммарная ёмкость школ, расположенных в квартале. Ёмкость 250 мест у всех 70 школ — модельное допущение: в OpenStreetMap атрибут вместимости отсутствует, поэтому это не подтверждённая вместимость.",
    "Большая ёмкость — в квартале или рядом есть школьные места.",
    "Ноль — школ в квартале нет; доступность мест оценивается через обеспеченность.",
    decimals=0,
    palette=BLUE,
    zero_separate=True,
    zero_label="0 — школ нет",
)
add_numeric(
    "school_count",
    "school_count",
    "Число школ",
    "шт.",
    "Школы и обеспеченность",
    "Количество школ в квартале после удаления дубликатов: из 82 школьных объектов OpenStreetMap осталось 70.",
    "Больше школ — выше насыщенность территории школами.",
    "Ноль — школ в границах квартала нет.",
    decimals=0,
    palette=BLUE,
    fixed_breaks=[0, 1, 2, 3],
)
add_numeric(
    "provision_strong",
    "provision_strong",
    "Обеспеченность школами (в пределах 15 минут)",
    "доля 0–1",
    "Школы и обеспеченность",
    "Доля нормативной потребности в школьных местах, покрытая школами в пределах 15 минут пешего пути (методика конкурентного распределения мест BlocksNet). Значение 0 — рядом нет доступных мест, 1 — потребность покрыта полностью.",
    "Ближе к 1 — потребность покрыта доступными школами.",
    "Ближе к 0 — дефицит мест в пределах норматива доступности.",
    decimals=3,
    palette=BLUE,
    fixed_breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
add_numeric(
    "provision_weak",
    "provision_weak",
    "Обеспеченность школами (с учётом дальних школ)",
    "доля 0–1",
    "Школы и обеспеченность",
    "Доля потребности с учётом ёмкости школ за пределами норматива доступности. В текущем расчёте оба показателя обеспеченности совпадают: школ дальше 15 минут не использовалось, поэтому ёмкость вне норматива равна нулю.",
    "Ближе к 1 — потребность покрыта даже с дальними школами.",
    "Ближе к 0 — дефицит сохраняется и с учётом дальних школ.",
    decimals=3,
    palette=BLUE,
    fixed_breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
add_numeric(
    "demand",
    "demand",
    "Нормативная потребность в местах",
    "мест",
    "Школы и обеспеченность",
    "Нормативная потребность квартала в школьных местах: 120 мест на 1000 жителей (норматив BlocksNet).",
    "Высокий спрос — квартал создаёт большую нагрузку на школьную сеть.",
    "Ноль — населения нет, потребность отсутствует.",
    decimals=0,
    palette=ORANGE,
    zero_separate=True,
    zero_label="0 — спроса нет",
)
add_numeric(
    "demand_within",
    "demand_within",
    "Спрос, покрытый в пределах норматива",
    "мест",
    "Школы и обеспеченность",
    "Часть нормативной потребности, которую удалось покрыть школами в пределах 15 минут.",
    "Больше — больше детей обеспечены близкими местами.",
    "Меньше — большая часть спроса остаётся вне нормативной доступности.",
    decimals=0,
    palette=ORANGE,
    zero_separate=True,
    zero_label="0 — покрытия нет",
)
add_numeric(
    "demand_left",
    "demand_left",
    "Непокрытая потребность (дефицит)",
    "мест",
    "Школы и обеспеченность",
    "Сколько нормативных мест осталось без покрытия после распределения доступной ёмкости между кварталами.",
    "Больше — острее дефицит школьных мест.",
    "Ноль — потребность покрыта полностью.",
    decimals=0,
    palette=ORANGE,
    zero_separate=True,
    zero_label="0 — дефицита нет",
)
add_numeric(
    "demand_without",
    "demand_without",
    "Спрос, покрытый только вне норматива",
    "мест",
    "Школы и обеспеченность",
    "Часть спроса, которую пришлось покрывать школами дальше нормативных 15 минут. В текущем расчёте все значения равны нулю.",
    "Больше — больше детей учатся далеко от дома.",
    "Ноль — вне нормативной доступности спрос не покрывался.",
    decimals=0,
    palette=ORANGE,
    zero_separate=True,
    zero_label="0 — все значения нулевые",
)
add_numeric(
    "capacity_within",
    "capacity_within",
    "Ёмкость, использованная в пределах норматива",
    "мест",
    "Школы и обеспеченность",
    "Школьная ёмкость, задействованная для кварталов в пределах 15 минут.",
    "Больше — больше мест реально работает на близлежащие кварталы.",
    "Ноль — близлежащая ёмкость не использовалась.",
    decimals=0,
    palette=BLUE,
    zero_separate=True,
    zero_label="0 — не использовалась",
)
add_numeric(
    "capacity_left",
    "capacity_left",
    "Остаток школьных мест",
    "мест",
    "Школы и обеспеченность",
    "Ёмкость, которая осталась невостребованной после распределения спроса.",
    "Больше — в квартале есть свободный резерв мест.",
    "Ноль — резерва нет.",
    decimals=0,
    palette=BLUE,
    zero_separate=True,
    zero_label="0 — резерва нет",
)
add_numeric(
    "capacity_without",
    "capacity_without",
    "Ёмкость, использованная вне норматива",
    "мест",
    "Школы и обеспеченность",
    "Школьная ёмкость, задействованная за пределами 15 минут. В текущем расчёте все значения равны нулю.",
    "Больше — больше мест работает на дальние кварталы.",
    "Ноль — вне норматива ёмкость не использовалась.",
    decimals=0,
    palette=BLUE,
    zero_separate=True,
    zero_label="0 — все значения нулевые",
)
add_numeric(
    "area_accessibility",
    "area_accessibility",
    "Пешеходная доступность территории",
    "мин",
    "Доступность",
    "Средневзвешенное по площади время пешего пути от квартала до всех остальных кварталов города, рассчитанное по матрице доступности пешеходного графа. Меньше — лучше.",
    "Больше минут — квартал менее связан с городом.",
    "Меньше минут — город доступнее из этого квартала.",
    decimals=1,
    palette=ORANGE,
)
add_numeric(
    "fsi",
    "fsi",
    "FSI — интенсивность застройки",
    "м²/м²",
    "Плотность и морфология",
    "Floor Space Index: суммарная поэтажная площадь зданий, делённая на площадь квартала. Показывает интенсивность использования территории.",
    "Выше FSI — плотнее и интенсивнее застройка.",
    "Ноль — застройки в квартале нет.",
    decimals=3,
    palette=PURPLE,
    zero_separate=True,
    zero_label="0 — застройки нет",
)
add_numeric(
    "gsi",
    "gsi",
    "GSI — доля территории под зданиями",
    "доля 0–1",
    "Плотность и морфология",
    "Ground Space Index: площадь пятен зданий, делённая на площадь квартала. Показывает, какая часть земли занята зданиями. Ноль означает, что на территории квартала зданий нет.",
    "Выше GSI — больше территории занято пятнами зданий.",
    "Ноль — зданий на территории нет.",
    decimals=3,
    palette=BROWN,
    zero_separate=True,
    zero_label="0 — зданий на территории нет",
)
add_numeric(
    "mxi",
    "mxi",
    "MXI — доля жилой площади",
    "доля 0–1",
    "Плотность и морфология",
    "Показатель рассчитывается как отношение жилой площади к суммарной поэтажной площади квартала. Это функциональный баланс жилого и нежилого, а не энтропийный индекс смешанности.",
    "Ближе к 1 — застройка преимущественно жилая.",
    "Ближе к 0 — застройка преимущественно нежилая.",
    decimals=3,
    palette=MAGENTA,
    fixed_breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
add_numeric(
    "l",
    "l",
    "L — средняя этажность",
    "этажи",
    "Плотность и морфология",
    "Средняя этажность застройки квартала: build_floor_area / footprint_area.",
    "Выше L — выше этажность.",
    "Ниже L — застройка более низкая.",
    decimals=1,
    palette=PURPLE,
)
add_numeric(
    "osr",
    "osr",
    "OSR — открытые пространства",
    "м²/м²",
    "Плотность и морфология",
    "Обеспеченность открытыми пространствами: (site_area − footprint_area) / build_floor_area. Отрицательные значения означают, что поэтажная площадь больше свободной площади квартала.",
    "Выше OSR — больше свободной площади на единицу застройки.",
    "Ниже OSR — территория более плотно занята застройкой.",
    decimals=2,
    palette=SLATE,
)
add_numeric(
    "share_living",
    "share_living",
    "Жилая площадь на единицу пятна",
    "м²/м²",
    "Плотность и морфология",
    "living_area / footprint_area: сколько жилой площади приходится на единицу площади пятна застройки.",
    "Выше — более компактная жилая застройка.",
    "Ноль — жилой площади нет.",
    decimals=2,
    palette=MAGENTA,
    zero_separate=True,
    zero_label="0 — жилой площади нет",
)
add_numeric(
    "share_non_living",
    "share_non_living",
    "Нежилая площадь на единицу пятна",
    "м²/м²",
    "Плотность и морфология",
    "non_living_area / footprint_area: сколько нежилой площади приходится на единицу площади пятна застройки.",
    "Выше — более компактная нежилая застройка.",
    "Ниже — меньше нежилой площади на пятно.",
    decimals=2,
    palette=BROWN,
)
add_numeric(
    "site_area_ha",
    "site_area_ha",
    "Площадь квартала",
    "га",
    "Геометрия",
    "Площадь квартала в гектарах (исходное значение в м²).",
    "Больше площадь — крупнее квартал.",
    "Меньше площадь — компактнее квартал.",
    decimals=2,
    palette=SLATE,
)
add_numeric(
    "footprint_area_ha",
    "footprint_area_ha",
    "Площадь застройки",
    "га",
    "Геометрия",
    "Площадь пятен зданий в гектарах (исходное значение в м²).",
    "Больше — больше земли занято зданиями.",
    "Ноль — застройки нет.",
    decimals=2,
    palette=BROWN,
    zero_separate=True,
    zero_label="0 — застройки нет",
)
add_numeric(
    "build_floor_area_ha",
    "build_floor_area_ha",
    "Суммарная поэтажная площадь",
    "га",
    "Геометрия",
    "Поэтажная площадь застройки в гектарах (исходное значение в м²).",
    "Больше — больше общий объём застройки.",
    "Ноль — застройки нет.",
    decimals=2,
    palette=PURPLE,
    zero_separate=True,
    zero_label="0 — застройки нет",
)
add_numeric(
    "living_area_ha",
    "living_area_ha",
    "Жилая площадь",
    "га",
    "Геометрия",
    "Жилая площадь в гектарах (исходное значение в м²).",
    "Больше — больше жилищного фонда.",
    "Ноль — жилой площади нет.",
    decimals=2,
    palette=MAGENTA,
    zero_separate=True,
    zero_label="0 — жилой площади нет",
)
add_numeric(
    "non_living_area_ha",
    "non_living_area_ha",
    "Нежилая площадь",
    "га",
    "Геометрия",
    "Нежилая площадь в гектарах (исходное значение в м²).",
    "Больше — больше нежилого фонда.",
    "Ноль — нежилой площади нет.",
    decimals=2,
    palette=BROWN,
    zero_separate=True,
    zero_label="0 — нежилой площади нет",
)
add_categorical(
    "land_use",
    "land_use",
    "Землепользование (преобладающее)",
    "Землепользование",
    "Преобладающий тип землепользования квартала. Категория «нет данных» означает, что однозначно определить преобладание не удалось.",
    LAND_USE_LABELS,
)
for column, title, description in [
    ("residential", "Доля жилой функции", "Доля площади квартала с жилой функцией."),
    ("business", "Доля деловой функции", "Доля площади квартала с деловой функцией."),
    ("recreation", "Доля рекреационной функции", "Доля площади квартала с рекреационной функцией."),
    ("industrial", "Доля промышленной функции", "Доля площади квартала с промышленной функцией."),
    ("transport", "Доля транспортной функции", "Доля площади квартала с транспортной функцией."),
    ("special", "Доля специальной функции", "Доля площади квартала со специальной функцией."),
    ("agriculture", "Доля сельскохозяйственной функции", "Доля площади квартала с сельскохозяйственной функцией."),
    ("share", "Доля преобладающей функции", "Доля площади, приходящаяся на преобладающий тип землепользования."),
]:
    add_numeric(
        column,
        column,
        title,
        "доля 0–1",
        "Землепользование",
        description,
        "Выше — функция занимает большую долю квартала.",
        "Ноль — эта функция в квартале не представлена.",
        decimals=3,
        palette=BLUE if column in ("residential", "business", "recreation") else ORANGE,
        zero_separate=column != "share",
        zero_label="0 — функции нет",
    )
add_categorical(
    "morphotype_spacematrix",
    "morphotype_spacematrix",
    "Морфотип Spacematrix",
    "Морфотипы",
    "Морфотип квартала по кластеризации FSI, MXI и L методом Spacematrix. Пропуск означает отсутствие застройки или недостаток данных.",
    SPACEMATRIX_LABELS,
)
add_categorical(
    "morphotype_strelka",
    "morphotype_strelka",
    "Морфотип Strelka",
    "Морфотипы",
    "Морфотип квартала по классификации Strelka.",
    STRELKA_LABELS,
)
add_categorical(
    "spacematrix_cluster_label",
    "spacematrix_cluster_label",
    "Кластер Spacematrix",
    "Морфотипы",
    "Номер кластера Spacematrix, к которому отнесён квартал.",
)

# Основной набор должен всегда содержать шесть обязательных показателей.
for required_metric in [
    "population",
    "provision_strong",
    "area_accessibility",
    "fsi",
    "gsi",
    "mxi",
]:
    if required_metric not in metric_config:
        fail(f"Не удалось собрать обязательный показатель: {required_metric}")

# Шесть обязательных показателей выводятся кнопками прямо над картой.
PRIMARY_METRICS = {
    "population": "Население",
    "provision_strong": "Обеспеченность",
    "area_accessibility": "Доступность",
    "fsi": "FSI",
    "gsi": "GSI",
    "mxi": "MXI",
}
for primary_metric in PRIMARY_METRICS:
    if primary_metric not in metric_config:
        fail(f"Не удалось собрать обязательный показатель: {primary_metric}")



# ---------------------------------------------------------------------------
# Сводные значения и проверки
# ---------------------------------------------------------------------------
checks = quality_report.get("checks", {})
checks_ok = all(bool(value) for value in checks.values() if isinstance(value, bool))
block_count = int(quality_report.get("blocks", len(ref_results)))
population = float(quality_report.get("population", ref_results["population"].fillna(0).sum()))
school_count = int(quality_report.get("schools", len(schools)))
capacity = float(quality_report.get("capacity", ref_results["capacity"].fillna(0).sum()))
max_area = float(blocks_quality.get("max_area_km2", ref_results["site_area"].max() / 1_000_000))
area_total = float(blocks_quality.get("blocks_area_km2", 0.0))
coverage_land = float(blocks_quality.get("coverage_land_pct", 0.0))
big_blocks = int(blocks_quality.get("blocks_gt_10_km2", 0))
developed_blocks = int((ref_results["fsi"].fillna(0) > 0).sum())


def summary_mean(column: str) -> float:
    try:
        value = float(results_summary.loc[column, "mean"])
        if np.isfinite(value):
            return value
    except (KeyError, TypeError, ValueError):
        pass
    return float(ref_results[column].replace([np.inf, -np.inf], np.nan).mean())


mean_provision = summary_mean("provision_strong")
mean_accessibility = summary_mean("area_accessibility")
top10_population_share = (
    float(ref_results["population"].fillna(0).nlargest(10).sum()) / population * 100.0
    if population
    else 0.0
)
populated = ref_results["population"].fillna(0) > 0
zero_provision_count = int(
    ((ref_results["provision_strong"].fillna(0) == 0) & populated).sum()
)
missing_provision_count = int(ref_results["provision_strong"].isna().sum())
provision_diff_max = float(
    (ref_results["provision_weak"] - ref_results["provision_strong"]).abs().max()
)
demand_total = float(provision_schools["demand"].fillna(0).sum())
demand_within_total = float(provision_schools["demand_within"].fillna(0).sum())
demand_left_total = float(provision_schools["demand_left"].fillna(0).sum())
capacity_left_total = float(provision_schools["capacity_left"].fillna(0).sum())
schools_modeled = int((schools["capacity_source"].astype(str) == "modeled_250").sum())
schools_osm = int((schools["capacity_source"].astype(str) == "osm").sum())
land_use_top = (
    ref_results["land_use"].fillna("нет данных").astype(str).value_counts().head(1)
)
top_land_use_text = land_use_top.index[0] if len(land_use_top) else "—"
morphotype_top = (
    ref_results["morphotype_spacematrix"].fillna("нет данных").astype(str).value_counts().head(1)
)
top_morphotype_text = morphotype_top.index[0] if len(morphotype_top) else "—"


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------
def digits_for_title(title: str) -> int:
    """Число знаков после запятой по единице измерения в заголовке столбца."""
    if any(mark in title for mark in ("м²", "чел", "мест", "шт")):
        return 0
    if "мин" in title:
        return 1
    return 3


def top_table(frame: pd.DataFrame, columns: dict[str, str]) -> str:
    display = pd.DataFrame(index=frame.index)
    display["ID квартала"] = frame.index
    for source, title in columns.items():
        display[title] = frame[source]
    for column in display.columns:
        if column != "ID квартала":
            digits = digits_for_title(column)
            display[column] = display[column].map(lambda value: fmt(value, digits))
    return table_html(display)


top_population = ref_results.sort_values("population", ascending=False).head(10).copy()
top_population_table = top_table(
    top_population,
    {
        "site_area": "Площадь, м²",
        "population": "Население, чел.",
        "capacity": "Ёмкость школ, мест",
        "provision_strong": "Обеспеченность, доля 0–1",
        "area_accessibility": "Доступность, мин",
    },
)

top_accessibility = (
    ref_results.dropna(subset=["area_accessibility"])
    .sort_values("area_accessibility", ascending=False)
    .head(10)
    .copy()
)
top_accessibility_table = top_table(
    top_accessibility,
    {
        "area_accessibility": "Доступность, мин",
        "population": "Население, чел.",
        "provision_strong": "Обеспеченность, доля 0–1",
        "site_area": "Площадь, м²",
    },
)

top_provision_low = (
    base[populated & base["provision_strong"].notna()]
    .sort_values(["provision_strong", "population"], ascending=[True, False])
    .head(10)
    .copy()
)
top_provision_low_table = top_table(
    top_provision_low,
    {
        "provision_strong": "Обеспеченность в пределах 15 минут, доля",
        "provision_weak": "Обеспеченность с учётом дальних школ, доля",
        "population": "Население, чел.",
        "capacity": "Ёмкость школ, мест",
        "demand_left": "Дефицит мест",
        "area_accessibility": "Доступность, мин",
    },
)

top_deficit = provision_schools.sort_values("demand_left", ascending=False).head(10).copy()
top_deficit_table = top_table(
    top_deficit,
    {
        "demand": "Спрос, мест",
        "capacity": "Ёмкость, мест",
        "demand_within": "Покрыто рядом, мест",
        "demand_left": "Дефицит, мест",
        "capacity_left": "Остаток мест",
    },
)

summary_display = results_summary.copy().reset_index()
summary_display = summary_display.rename(columns={summary_display.columns[0]: "Показатель"})
summary_display["Показатель"] = summary_display["Показатель"].map(
    lambda column: METRIC_TITLES.get(column, (column, ""))[0]
)
summary_display["Единица"] = summary_display["Показатель"].map(
    lambda name: next(
        (unit for title, unit in METRIC_TITLES.values() if title == name),
        "",
    )
)
summary_display = summary_display.rename(columns=SUMMARY_COLUMNS)
ordered_columns = ["Показатель", "Единица"] + [
    SUMMARY_COLUMNS[column] for column in results_summary.columns if column in SUMMARY_COLUMNS
]
summary_display = summary_display[[column for column in ordered_columns if column in summary_display.columns]]
for column in summary_display.columns:
    if column in ("Показатель", "Единица"):
        continue
    digits = 0 if column == SUMMARY_COLUMNS["count"] else 3
    summary_display[column] = summary_display[column].map(lambda value: fmt(value, digits))
summary_table = table_html(summary_display)

numeric_rows = []
for column in sorted(ref_results.select_dtypes(include=[np.number]).columns):
    raw = pd.to_numeric(ref_results[column], errors="coerce")
    infinite = int(np.isinf(raw.to_numpy(dtype="float64", na_value=np.nan)).sum())
    series = raw.replace([np.inf, -np.inf], np.nan)
    title, unit = METRIC_TITLES.get(column, (column, "число"))
    numeric_rows.append(
        {
            "Показатель": title,
            "Единица": unit,
            "Среднее": fmt(series.mean(), 3),
            "Медиана": fmt(series.median(), 3),
            "Минимум": fmt(series.min(), 3),
            "Максимум": fmt(series.max(), 3),
            "Нулей": fmt(int((series == 0).sum()), 0),
            "Нет данных": fmt(int(series.isna().sum()), 0),
            "Бесконечных": fmt(infinite, 0),
            "Кварталов": fmt(int(series.notna().sum()), 0),
        }
    )
numeric_table = table_html(pd.DataFrame(numeric_rows))

categorical_rows = []
CATEGORY_LABELS = {
    "land_use": LAND_USE_LABELS,
    "morphotype_spacematrix": SPACEMATRIX_LABELS,
    "morphotype_strelka": STRELKA_LABELS,
}
for column, title in [
    ("land_use", "Землепользование"),
    ("morphotype_spacematrix", "Морфотип Spacematrix"),
    ("morphotype_strelka", "Морфотип Strelka"),
    ("spacematrix_cluster_label", "Кластер Spacematrix"),
]:
    source_series = base[column] if column == "spacematrix_cluster_label" else ref_results[column]
    series = source_series.where(source_series.notna(), None)
    counts = series.dropna().astype(str).value_counts()
    labels = CATEGORY_LABELS.get(column, {})
    total = int(counts.sum()) if len(counts) else 0
    for category, count in counts.items():
        categorical_rows.append(
            {
                "Показатель": title,
                "Категория": labels.get(category, category),
                "Кварталов": fmt(int(count), 0),
                "Доля, %": fmt(100.0 * int(count) / total if total else 0.0, 1),
            }
        )
categorical_table = table_html(pd.DataFrame(categorical_rows))

check_labels = {
    "data_dir_refined": "Каталог данных",
    "blocks_present": "Кварталы присутствуют",
    "all_geometries_valid": "Все геометрии валидны",
    "unique_block_ids": "ID кварталов уникальны",
    "matrix_shape": "Размер матрицы корректен",
    "matrix_index": "Индексы матрицы совпадают",
    "matrix_diagonal_zero": "Диагональ матрицы нулевая",
    "matrix_no_nan": "В матрице нет NaN",
    "population_positive": "Население положительно",
    "school_capacity_positive": "Ёмкость школ положительна",
    "results_rows_match": "Число строк результатов совпадает",
    "results_columns_match": "Набор колонок результатов совпадает",
    "all_required_metrics": "Все обязательные метрики присутствуют",
    "land_use_present": "Землепользование заполнено",
    "same_columns_as_original": "Структура колонок согласована",
}
quality_rows = [
    {"Проверка": check_labels.get(key, key), "Результат": "OK" if bool(value) else "НЕТ"}
    for key, value in checks.items()
]
quality_table = table_html(pd.DataFrame(quality_rows))

QUALITY_META_LABELS = {
    "blocks": "Кварталов в расчёте",
    "population": "Модельное население, чел.",
    "schools": "Школ после удаления дубликатов",
    "capacity": "Суммарная модельная вместимость школ, мест",
    "matrix_shape": "Размер матрицы доступности",
    "missing_metrics": "Показатели без значений",
}
BLOCKS_QUALITY_LABELS = {
    "blocks_before_split": "Кварталов до разделения крупных",
    "blocks_after_split": "Кварталов после разделения крупных",
    "blocks_area_km2": "Суммарная площадь кварталов, км²",
    "area_change_km2": "Изменение площади при разделении, км²",
    "land_area_km2": "Площадь территории, км²",
    "coverage_land_pct": "Покрытие территории кварталами, %",
    "max_area_km2": "Максимальная площадь квартала, км²",
    "blocks_gt_10_km2": "Кварталов площадью более 10 км²",
    "blocks_holes_gt_50": "Кварталов с внутренними пустотами",
    "invalid_geometries": "Невалидных геометрий",
    "overlap_area_km2": "Площадь пересечений кварталов, км²",
    "outside_area_km2": "Площадь кварталов вне границы города, км²",
    "top4_area_share_pct": "Доля четырёх крупнейших кварталов в площади, %",
}
MATRIX_META_LABELS = {
    "shape": "Размер матрицы (квадратов)",
    "chunk_size": "Размер порции расчёта",
    "workers": "Параллельных процессов",
    "graph_nodes": "Узлов пешеходного графа",
    "graph_edges": "Рёбер пешеходного графа",
    "units": "Единицы времени",
    "dtype": "Тип данных матрицы",
    "parts": "Порций в матрице",
    "blocks_signature": "Отпечаток состава кварталов",
    "created_at_utc": "Расчёт завершён (UTC)",
}

quality_meta_rows = [
    {
        "Параметр": QUALITY_META_LABELS.get(key, key),
        "Значение": fmt(value, 3) if not isinstance(value, (list, dict)) else html.escape(str(value)),
    }
    for key, value in quality_report.items()
    if key != "checks"
]
quality_meta_table = table_html(pd.DataFrame(quality_meta_rows))

blocks_quality_table = table_html(
    pd.DataFrame(
        [
            {
                "Показатель": BLOCKS_QUALITY_LABELS.get(key, key),
                "Значение": fmt(value, 3),
            }
            for key, value in blocks_quality.items()
        ]
    )
)
MATRIX_META_VALUES = {
    "units": "минуты пешего передвижения",
    "dtype": "числа половинной точности (float16)",
}
matrix_meta_table = table_html(
    pd.DataFrame(
        [
            {
                "Параметр": MATRIX_META_LABELS.get(key, key),
                "Значение": MATRIX_META_VALUES.get(
                    key,
                    fmt(value, 3) if not isinstance(value, (list, dict)) else html.escape(str(value)),
                ),
            }
            for key, value in matrix_meta.items()
        ]
    )
)

# Источник данных, параметры нарезки и диагностика геометрии по кварталам.
OSM_SOURCE_LABELS = {
    "source": "Источник данных",
    "url": "Адрес архива",
    "license": "Лицензия",
    "note": "Примечание",
    "buffer_km": "Буфер вокруг территории, км",
    "bbox_4326": "Границы выгрузки (долгота, широта)",
    "cropped_pbf": "Локальный файл",
    "cropped_size_bytes": "Размер локального файла, байт",
    "boundary_osm_id": "Граница города в OpenStreetMap",
}
OSM_SOURCE_VALUES = {
    "note": "локальный PBF; загрузка данных в ноутбуке не выполняется",
    "source": "Уральский федеральный округ (Geofabrik), локальный PBF",
}
source_meta_table = table_html(
    pd.DataFrame(
        [
            {
                "Параметр": OSM_SOURCE_LABELS.get(key, key),
                "Значение": html.escape(
                    OSM_SOURCE_VALUES.get(
                        key,
                        fmt(value, 3) if not isinstance(value, (list, dict)) else str(value),
                    )
                ),
            }
            for key, value in osm_source.items()
        ]
    )
)

SPLIT_META_LABELS = {
    "method": "Метод разделения крупных кварталов",
    "cluster_backend": "Библиотека кластеризации",
    "max_area_km2": "Порог площади квартала, км²",
    "min_block_area_m2": "Минимальная площадь квартала, м²",
    "input_blocks": "Кварталов до разделения",
    "output_blocks": "Кварталов после разделения",
    "input_signature": "Отпечаток исходной структуры",
    "output_signature": "Отпечаток итоговой структуры",
    "created_at_utc": "Расчёт завершён (UTC)",
    "note": "Примечание",
}
SPLIT_META_VALUES = {
    "method": "area_classifier — разделение квартала по площади и плотности застройки",
    "cluster_backend": "sklearn.HDBSCAN — иерархическая кластеризация точек (замена hdbscan, недоступного в окружении)",
}
split_params = split_meta.get("params", {})
split_meta_rows = [
    {
        "Параметр": SPLIT_META_LABELS.get(key, key),
        "Значение": SPLIT_META_VALUES.get(key, fmt(value, 3)),
    }
    for key, value in split_params.items()
]
split_meta_rows += [
    {"Параметр": SPLIT_META_LABELS.get(key, key), "Значение": fmt(value, 3)}
    for key, value in split_meta.items()
    if key != "params"
]
split_meta_table = table_html(pd.DataFrame(split_meta_rows))

blocks_quality_stats = pd.DataFrame(
    [
        {"Показатель": "Кварталов в диагностике", "Значение": fmt(len(blocks_quality_rows), 0)},
        {
            "Показатель": "Средняя площадь квартала, км²",
            "Значение": fmt(blocks_quality_rows["area_km2"].mean(), 4),
        },
        {
            "Показатель": "Медианная площадь квартала, км²",
            "Значение": fmt(blocks_quality_rows["area_km2"].median(), 4),
        },
        {
            "Показатель": "Максимальная площадь квартала, км²",
            "Значение": fmt(blocks_quality_rows["area_km2"].max(), 4),
        },
        {
            "Показатель": "Кварталов с внутренними пустотами",
            "Значение": fmt(int((blocks_quality_rows["holes"] > 0).sum()), 0),
        },
        {
            "Показатель": "Максимум внутренних пустот в квартале",
            "Значение": fmt(int(blocks_quality_rows["holes"].max()), 0),
        },
        {
            "Показатель": "Среднее число вершин в контуре",
            "Значение": fmt(blocks_quality_rows["vertices"].mean(), 1),
        },
        {
            "Показатель": "Средняя компактность контура, 0–1",
            "Значение": fmt(blocks_quality_rows["compactness"].mean(), 3),
        },
    ]
)
blocks_quality_stats_table = table_html(blocks_quality_stats)

PROVISION_STAT_LABELS = {
    "demand": ("Нормативная потребность", "мест"),
    "demand_within": ("Покрыто в пределах 15 минут", "мест"),
    "demand_without": ("Покрыто только дальними школами", "мест"),
    "demand_left": ("Непокрытая потребность (дефицит)", "мест"),
    "capacity": ("Ёмкость школ", "мест"),
    "capacity_within": ("Ёмкость, использованная в пределах 15 минут", "мест"),
    "capacity_without": ("Ёмкость, использованная вне норматива", "мест"),
    "capacity_left": ("Остаток школьных мест", "мест"),
    "provision_strong": ("Обеспеченность в пределах 15 минут", "доля 0–1"),
    "provision_weak": ("Обеспеченность с учётом дальних школ", "доля 0–1"),
}
PROVISION_STAT_COLUMNS = {
    "count": "Кварталов",
    "mean": "Среднее",
    "std": "Стандартное отклонение",
    "min": "Минимум",
    "median": "Медиана",
    "max": "Максимум",
}
provision_stats = provision_schools.describe().T.reset_index()
provision_stats = provision_stats.rename(columns={"index": "Показатель", "50%": "median"})
provision_stats["Показатель"] = provision_stats["Показатель"].map(
    lambda column: PROVISION_STAT_LABELS.get(column, (column, ""))[0]
)
provision_stats["Единица"] = provision_stats["Показатель"].map(
    lambda name: next((unit for title, unit in PROVISION_STAT_LABELS.values() if title == name), "")
)
provision_stats = provision_stats.rename(columns=PROVISION_STAT_COLUMNS)
for column in ["Кварталов", "Среднее", "Стандартное отклонение", "Минимум", "Медиана", "Максимум"]:
    if column in provision_stats.columns:
        digits = 0 if column == "Кварталов" else 3
        provision_stats[column] = provision_stats[column].map(lambda value: fmt(value, digits))
provision_table = table_html(provision_stats)

provision_totals = pd.DataFrame(
    [
        {
            "Показатель": "Нормативная потребность, мест (сумма по кварталам)",
            "Значение": fmt(demand_total, 0),
        },
        {
            "Показатель": "Покрыто в пределах 15 минут, мест (сумма по кварталам)",
            "Значение": fmt(demand_within_total, 0),
        },
        {
            "Показатель": "Непокрытая потребность, мест (сумма по кварталам)",
            "Значение": fmt(demand_left_total, 0),
        },
        {
            "Показатель": "Остаток школьных мест, мест (сумма по кварталам)",
            "Значение": fmt(capacity_left_total, 0),
        },
        {
            "Показатель": "Средняя обеспеченность в пределах 15 минут (среднее по кварталам)",
            "Значение": fmt(mean_provision, 3),
        },
        {
            "Показатель": "Средняя обеспеченность с учётом дальних школ (среднее по кварталам)",
            "Значение": fmt(summary_mean("provision_weak"), 3),
        },
    ]
)
provision_totals_table = table_html(provision_totals)

schools_display = schools[["name", "capacity", "capacity_source"]].copy()
schools_display["name"] = schools_display["name"].fillna("Без названия")
schools_display["capacity_source"] = schools_display["capacity_source"].map(
    {"modeled_250": "модельная, 250 мест", "osm": "из OSM"}
).fillna("не указан")
schools_display = schools_display.rename(
    columns={
        "name": "Школа",
        "capacity": "Ёмкость, мест",
        "capacity_source": "Источник ёмкости",
    }
)
schools_display["Ёмкость, мест"] = schools_display["Ёмкость, мест"].map(lambda value: fmt(value, 0))
schools_table = table_html(schools_display)

links_display = provision_links.sort_values("value", ascending=False).head(20).copy()
links_display = links_display.rename(
    columns={
        "source": "Квартал-потребитель",
        "target": "Квартал школы",
        "value": "Передано мест",
    }
)
links_display["Передано мест"] = links_display["Передано мест"].map(lambda value: fmt(value, 0))
links_table = table_html(links_display)

clusters_display = spacematrix_clusters.copy()
for column in ["l", "fsi", "mxi"]:
    if column in clusters_display.columns:
        clusters_display[column] = clusters_display[column].map(lambda value: fmt(value, 3))
clusters_table = table_html(clusters_display)

# ---------------------------------------------------------------------------
# График выбранного показателя: панель аналитики справа от карты
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def axis_number(value: float) -> str:
    """Подпись оси: без дробей для целых, с разрядами через пробел."""
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", "\u00a0")
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(axis=grid_axis, color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d4d4d4")
    ax.tick_params(colors="#525252", labelsize=9)
    ax.title.set_color("#171717")
    ax.xaxis.label.set_color("#262626")
    ax.yaxis.label.set_color("#262626")


def metric_chart(key: str, cfg: dict) -> dict:
    """Гистограмма для числового показателя и столбчатая диаграмма для категории.

    График строится один раз для каждого показателя и подставляется в панель
    аналитики при переключении показателя на карте.
    """
    label = str(cfg["label"])
    unit = str(cfg["unit"])
    figure, axis = plt.subplots(figsize=(5.0, 3.4))

    if cfg["type"] == "categorical":
        categories = list(cfg["categories"])
        names = [str(item["label"]) for item in categories]
        counts = [int(item["count"]) for item in categories]
        colors = [str(item["color"]) for item in categories]
        positions = np.arange(len(names))
        axis.barh(positions, counts, color=colors, height=0.66)
        axis.set_yticks(positions)
        axis.set_yticklabels([fill(name, 32) for name in names])
        axis.invert_yaxis()
        axis.set_xlabel("Кварталов")
        axis.set_title(f"Категории показателя «{label}»")
        for position, count in zip(positions, counts):
            axis.text(
                count,
                position,
                "  " + fmt(count, 0),
                va="center",
                fontsize=8.5,
                color="#525252",
            )
        axis.set_xlim(0, max(counts) * 1.18 if max(counts) else 1)
        style_axis(axis, grid_axis="x")
        figure.tight_layout()
        caption = (
            f"Категориальный показатель: {len(names)} "
            f"{plural(len(names), 'категория', 'категории', 'категорий')}, "
            f"{fmt(cfg['n'], 0)} "
            f"{plural(cfg['n'], 'квартал', 'квартала', 'кварталов')} с определённой категорией."
        )
        if cfg["missing_count"]:
            caption += (
                f" Без определённой категории — {fmt(cfg['missing_count'], 0)} "
                f"{plural(cfg['missing_count'], 'квартал', 'квартала', 'кварталов')}."
            )
        return {"src": image_data_uri(figure), "caption": caption, "title": label}

    values = pd.to_numeric(
        base.get(cfg["key"], pd.Series(dtype="float64")), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    zeros = int((valid == 0).sum())
    missing = int(values.isna().sum())
    plotted = valid[valid > 0] if zeros else valid
    if plotted.empty:
        plotted = valid

    palette = list(cfg["colors"]) or ["#9fc5f2"]
    color = palette[min(len(palette) // 2, len(palette) - 1)]

    if cfg["type"] == "constant":
        axis.bar([0], [cfg["n"] or 0], color=color, width=0.5)
        axis.set_xticks([0])
        axis.set_xticklabels([fmt(cfg["constant_value"], cfg["decimals"])])
        axis.set_xlabel(f"{label}, {unit}")
        axis.set_ylabel("Кварталов")
        axis.set_title(f"Распределение показателя «{label}»")
        style_axis(axis)
        figure.tight_layout()
        caption = (
            f"Все сохранённые значения одинаковы: {fmt(cfg['constant_value'], cfg['decimals'])} {unit}."
        )
        return {"src": image_data_uri(figure), "caption": caption, "title": label}

    median = float(plotted.median()) if len(plotted) else 0.0
    use_log = False
    if len(plotted) > 50 and median > 0 and float(plotted.max()) / median > 40:
        use_log = True

    data = np.log1p(plotted.to_numpy()) if use_log else plotted.to_numpy()
    bins = 36 if len(data) > 200 else max(8, int(len(data) ** 0.5))
    axis.hist(data, bins=bins, color=color, edgecolor="#ffffff", linewidth=0.4)
    if use_log:
        ticks = [0, 1, 2, 3, 4]
        axis.set_xticks(ticks)
        axis.set_xticklabels([fmt(10 ** tick - 1, 0) for tick in ticks])
        axis.set_xlabel(f"{label}, {unit} (логарифмическая шкала)")
    else:
        axis.set_xlabel(f"{label}, {unit}")
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, position: axis_number(value))
        )
    axis.set_ylabel("Кварталов")
    axis.set_title(f"Распределение показателя «{label}»")
    median_position = np.log1p(median) if use_log else median
    axis.axvline(median_position, color="#171717", linestyle="--", linewidth=1.1)
    style_axis(axis)
    figure.tight_layout()

    caption = (
        f"Распределение по {fmt(len(plotted), 0)} "
        f"{plural(len(plotted), 'кварталу', 'кварталам', 'кварталам')} "
        + ("с ненулевыми значениями; " if zeros else "со значениями; ")
        + f"медиана — {fmt(median, cfg['decimals'])} {unit} (пунктир)."
    )
    if use_log:
        caption += " Шкала логарифмическая: разброс значений велик, поэтому ось сжата."
    if zeros:
        caption += (
            f" Отдельно учтены нули — {fmt(zeros, 0)} "
            f"{plural(zeros, 'квартал', 'квартала', 'кварталов')}: на карте это светло-серый класс."
        )
    if missing:
        caption += (
            f" Значения не применимы или отсутствуют у {fmt(missing, 0)} "
            f"{plural(missing, 'квартала', 'кварталов', 'кварталов')}: на карте они остаются белыми."
        )
    return {"src": image_data_uri(figure), "caption": caption, "title": label}


metric_charts: dict[str, dict] = {}
for metric_key, metric_cfg in metric_config.items():
    metric_charts[metric_key] = metric_chart(metric_key, metric_cfg)


# ---------------------------------------------------------------------------
# Соответствие исходному заданию: показатель — ноутбук — данные — страница
# ---------------------------------------------------------------------------
coverage_rows = [
    (
        "Матрица доступности: время пешего пути между кварталами",
        "да — раздел «Доступность и сеть»",
        "да — 12 805 × 12 805, 513 порций",
        "частично — сама матрица не встраивается из-за размера, показаны производные",
        "представлено производными величинами",
    ),
    (
        "Население квартала",
        "да — «Здания и население»",
        "да — модельная оценка по жилой площади",
        "да — карта, график, таблицы, выводы",
        "показано",
    ),
    (
        "Площадь пятен застройки",
        "да — «Плотность и застройка»",
        "да",
        "да — карта (в га), таблицы",
        "показано",
    ),
    (
        "Суммарная поэтажная площадь",
        "да — «Плотность и застройка»",
        "да",
        "да — карта (в га), таблицы",
        "показано",
    ),
    (
        "Жилая и нежилая площадь",
        "да — «Плотность и застройка»",
        "да",
        "да — карта (в га), таблицы",
        "показано",
    ),
    (
        "Число зданий в квартале",
        "да — «Здания и население»",
        "да",
        "да — карта, таблицы",
        "показано",
    ),
    (
        "Число школ в квартале",
        "да — «Школы»",
        "да — 70 школ после удаления дубликатов",
        "да — карта, таблицы",
        "показано",
    ),
    (
        "Вместимость школ",
        "да — «Школы»",
        "да — модельная: 250 мест на школу",
        "да — карта, таблицы; везде помечено как модельная оценка",
        "показано как модельная оценка",
    ),
    (
        "Доли землепользования и преобладающая функция",
        "да — «Землепользование»",
        "да",
        "да — карта категорий, таблицы",
        "показано",
    ),
    (
        "Нормативная потребность в школьных местах",
        "да — «Обеспеченность услугами»",
        "да — 120 мест на 1000 жителей",
        "да — карта, таблицы",
        "показано",
    ),
    (
        "Покрытие потребности в пределах норматива и за его пределами",
        "да — «Обеспеченность услугами»",
        "да",
        "да — карта, таблицы",
        "показано",
    ),
    (
        "Дефицит и остаток школьных мест",
        "да — «Обеспеченность услугами»",
        "да",
        "да — карта, таблицы, выводы",
        "показано",
    ),
    (
        "Обеспеченность школами: оба показателя",
        "да — «Обеспеченность услугами»",
        "да",
        "да — карта, график, таблицы",
        "показано",
    ),
    (
        "Распределение спроса между школами (связки «квартал — школа»)",
        "да — «Обеспеченность услугами»",
        "да",
        "да — таблица связок",
        "показано",
    ),
    (
        "Плотностные показатели: FSI, GSI, MXI, L, OSR, доли жилой и нежилой площади",
        "да — «Плотность и застройка»",
        "да",
        "да — карта, графики, таблицы",
        "показано",
    ),
    (
        "Типы застройки Spacematrix",
        "да — «Морфотипы застройки»",
        "да",
        "да — карта категорий, таблица, портреты кластеров",
        "показано",
    ),
    (
        "Классификация «Стрелки» (Strelka)",
        "да — «Морфотипы застройки»",
        "да",
        "да — карта категорий, таблица",
        "показано",
    ),
    (
        "Доступность с учётом площади кварталов",
        "да — «Доступность и сеть»",
        "да",
        "да — карта, график, таблицы",
        "показано",
    ),
    (
        "Среднее, медианное и максимальное время доступности",
        "нет — только описаны в справочнике метрик",
        "нет — отдельные колонки не сохранены",
        "нет",
        "отсутствует: требует отдельного расчёта",
    ),
    (
        "Относительная доступность, связность, центральность",
        "нет — только описаны в справочнике метрик",
        "нет",
        "нет",
        "отсутствует: требует отдельного расчёта",
    ),
    (
        "Вакантные (незастроенные) территории",
        "нет — раздел описан, но не рассчитан",
        "нет",
        "нет",
        "отсутствует: требует отдельного расчёта",
    ),
    (
        "Услуги кроме школ и разнообразие услуг",
        "нет — только описаны в справочнике метрик",
        "нет",
        "нет",
        "отсутствует: требует отдельного расчёта",
    ),
    (
        "Матрица расстояний, граф смежности, потоки «откуда — куда», загруженность дорог",
        "нет — только описаны в справочнике метрик",
        "нет",
        "нет",
        "отсутствует: требует отдельного расчёта",
    ),
    (
        "Совместимость и сходство землепользований, предсказание типа землепользования",
        "нет — только описаны в справочнике метрик",
        "нет",
        "нет",
        "отсутствует: требует отдельного расчёта",
    ),
]
coverage_table = table_html(
    pd.DataFrame(
        [
            {
                "Показатель": item,
                "Есть в ноутбуке": notebook,
                "Есть в данных": data,
                "Есть на странице": page,
                "Статус": status,
            }
            for item, notebook, data, page, status in coverage_rows
        ]
    )
)


# ---------------------------------------------------------------------------
# Данные одной интерактивной карты
# ---------------------------------------------------------------------------
MAP_FIELDS = [
    "id",
    "area_ha",
    "area_km2",
    "population",
    "population_density",
    "building_count",
    "school_count",
    "capacity",
    "provision_strong",
    "provision_weak",
    "demand",
    "demand_within",
    "demand_without",
    "demand_left",
    "capacity_within",
    "capacity_without",
    "capacity_left",
    "area_accessibility",
    "fsi",
    "gsi",
    "mxi",
    "l",
    "osr",
    "share_living",
    "share_non_living",
    "site_area_ha",
    "footprint_area_ha",
    "build_floor_area_ha",
    "living_area_ha",
    "non_living_area_ha",
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
    "share",
    "land_use",
    "morphotype_spacematrix",
    "morphotype_strelka",
    "spacematrix_cluster_label",
]
for field in MAP_FIELDS:
    if field not in base.columns:
        base[field] = None

map_gdf = base[["geometry", *MAP_FIELDS]].copy()
map_gdf = map_gdf.to_crs(utm)
map_gdf["geometry"] = map_gdf.geometry.simplify(7, preserve_topology=True)
map_gdf = map_gdf.to_crs(4326)
for column in MAP_FIELDS:
    if column == "id":
        continue
    if pd.api.types.is_numeric_dtype(map_gdf[column]):
        map_gdf[column] = pd.to_numeric(map_gdf[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        map_gdf[column] = map_gdf[column].round(4)
    else:
        map_gdf[column] = map_gdf[column].where(map_gdf[column].notna(), None)

block_data = json.loads(map_gdf.to_json(na="null", drop_id=True))

water_map = water.to_crs(4326)
water_geojson = json.loads(water_map.to_json(drop_id=True))
# Размер каждого водного объекта: на обзорном масштабе показываем только крупные,
# мелкие — при приближении. Размер считается в метрической проекции, геоданные не меняются.
water_utm = water.to_crs(utm)
water_sizes: list[tuple[float, float]] = []
for geometry in water_utm.geometry:
    if geometry is None or geometry.is_empty:
        water_sizes.append((0.0, 0.0))
    elif geometry.geom_type in ("Polygon", "MultiPolygon"):
        water_sizes.append((float(geometry.area) / 1_000_000, 0.0))
    else:
        water_sizes.append((0.0, float(geometry.length) / 1000))
for feature, (area_km2, length_km) in zip(water_geojson["features"], water_sizes):
    feature.setdefault("properties", {})
    feature["properties"]["area_km2"] = round(area_km2, 6)
    feature["properties"]["length_km"] = round(length_km, 4)
water_data = water_geojson
boundary_map = boundaries.to_crs(4326)
boundary_data = json.loads(boundary_map.to_json(drop_id=True))

schools_map = schools.to_crs(4326).copy()
school_features = []
for _, row in schools_map.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    point = geom if geom.geom_type == "Point" else geom.representative_point()
    school_features.append(
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(point.x), float(point.y)]},
            "properties": {
                "name": row.get("name") if pd.notna(row.get("name")) else "Школа без названия",
                "capacity": float(row.get("capacity")) if pd.notna(row.get("capacity")) else None,
                "capacity_source": str(row.get("capacity_source", "не указан")),
            },
        }
    )
school_data = {"type": "FeatureCollection", "features": school_features}

if len(boundary_map):
    minx, miny, maxx, maxy = boundary_map.total_bounds
    boundary_bounds = [[float(miny), float(minx)], [float(maxy), float(maxx)]]
else:
    minx, miny, maxx, maxy = map_gdf.total_bounds
    boundary_bounds = [[float(miny), float(minx)], [float(maxy), float(maxx)]]

object_meta = {
    "water": int(len(water)),
    "schools": int(len(schools)),
    "boundary": int(len(boundaries)),
    "blocks": int(len(map_gdf)),
    "outline_color": OUTLINE_COLOR,
    "water_color": WATER_COLOR,
}

# ---------------------------------------------------------------------------
# Один Leaflet-инстанс; все метрики переключаются стилями GeoJSON
# ---------------------------------------------------------------------------
MAP_SCRIPT_TEMPLATE = r"""
<script>
(function () {
  "use strict";

  var mapElement = document.getElementById("tyumen-map");
  if (!window.L) {
    mapElement.innerHTML = '<div class="map-fallback">Не удалось загрузить библиотеку Leaflet с CDN. ' +
      'Проверьте подключение к интернету: HTML-файл содержит все геоданные, но для интерактивной карты нужны Leaflet CSS и JS.</div>';
    return;
  }

  var metrics = __METRICS__;
  var metricCharts = __METRIC_CHARTS__;
  var blockData = __BLOCK_DATA__;
  var waterData = __WATER_DATA__;
  var boundaryData = __BOUNDARY_DATA__;
  var schoolData = __SCHOOL_DATA__;
  var objectMeta = __OBJECT_META__;
  var boundaryBounds = __BOUNDARY_BOUNDS__;

  var selected = "population";
  if (!metrics[selected]) {
    selected = Object.keys(metrics)[0];
  }
  var outlinesOn = true;
  var activePopupLayer = null;
  var activeTooltipLayer = null;

  var map = L.map("tyumen-map", {
    attributionControl: false,
    zoomControl: true,
    preferCanvas: true,
    minZoom: 9,
    maxZoom: 20
  });
  map.setView([57.1522, 65.5412], 10);

  // Порядок слоёв: кварталы → вода (контур) → школы → граница города.
  map.createPane("blocksPane");
  map.getPane("blocksPane").style.zIndex = "400";
  map.createPane("waterPane");
  map.getPane("waterPane").style.zIndex = "550";
  map.createPane("schoolsPane");
  map.getPane("schoolsPane").style.zIndex = "600";
  map.createPane("boundaryPane");
  map.getPane("boundaryPane").style.zIndex = "700";

  function esc(value) {
    return String(value === null || value === undefined ? "" : value).replace(/[&<>"']/g, function (char) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char];
    });
  }

  function fmtNumber(value, decimals) {
    if (value === null || value === undefined || value === "") return "—";
    var number = Number(value);
    if (!isFinite(number)) return "—";
    return number.toLocaleString("ru-RU", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }

  function fmt(value, decimals) {
    return fmtNumber(value, decimals);
  }

  function unitText(cfg) {
    return cfg.unit === "доля 0–1" ? "доля" : (cfg.unit || "");
  }

  function valueFor(properties, cfg) {
    if (!cfg) return null;
    var raw = properties[cfg.key];
    if (raw === null || raw === undefined || raw === "") return null;
    if (cfg.type === "categorical") return String(raw);
    var number = Number(raw);
    return isFinite(number) ? number : null;
  }

  function findCategory(cfg, value) {
    if (!cfg || !cfg.categories || value === null || value === undefined) return null;
    var wanted = String(value);
    for (var i = 0; i < cfg.categories.length; i += 1) {
      if (String(cfg.categories[i].value) === wanted) return cfg.categories[i];
    }
    return null;
  }

  function categoryLabel(metricKey, value) {
    var category = findCategory(metrics[metricKey], value);
    return category ? category.label : "нет данных";
  }

  function classIndex(value, cfg) {
    var breaks = cfg.breaks || [];
    if (breaks.length < 2) return 0;
    for (var i = 1; i < breaks.length; i += 1) {
      if (Number(value) < breaks[i]) return i - 1;
    }
    return breaks.length - 2;
  }

  function colorFor(value, cfg) {
    if (!cfg) return "#bdbdbd";
    if (cfg.type === "categorical") {
      if (value === null || value === undefined) return cfg.missing_color;
      var category = findCategory(cfg, value);
      return category ? category.color : cfg.missing_color;
    }
    if (value === null || value === undefined) return cfg.missing_color;
    if (cfg.type === "constant") return cfg.colors[0] || "#bdbdbd";
    if (cfg.zero_separate && Number(value) === 0) return cfg.zero_color;
    var index = classIndex(value, cfg);
    return cfg.colors[Math.min(index, cfg.colors.length - 1)] || "#bdbdbd";
  }

  function classCounts(cfg) {
    var counts = [];
    var i;
    if (!cfg || cfg.type === "categorical") return counts;
    for (i = 0; i < cfg.colors.length; i += 1) counts.push(0);
    var layers = thematicLayer.getLayers();
    for (i = 0; i < layers.length; i += 1) {
      var props = layers[i].feature ? layers[i].feature.properties : null;
      if (!props) continue;
      var value = valueFor(props, cfg);
      if (value === null) continue;
      if (cfg.type === "constant") { counts[0] += 1; continue; }
      if (cfg.zero_separate && Number(value) === 0) continue;
      var index = classIndex(value, cfg);
      if (index >= 0 && index < counts.length) counts[index] += 1;
    }
    return counts;
  }

  function valueLabel(properties, metricKey) {
    var cfg = metrics[metricKey];
    var value = valueFor(properties, cfg);
    if (value === null) return "нет данных";
    if (cfg.type === "categorical") {
      return categoryLabel(metricKey, value);
    }
    if (cfg.type === "constant") {
      return fmtNumber(cfg.constant_value, cfg.decimals) + (unitText(cfg) ? " " + unitText(cfg) : "");
    }
    return fmtNumber(value, cfg.decimals) + (unitText(cfg) ? " " + unitText(cfg) : "");
  }

  function cell(label, value) {
    return "<tr><th>" + esc(label) + "</th><td>" + esc(value) + "</td></tr>";
  }

  function tooltipHtml(properties) {
    var cfg = metrics[selected];
    var area = fmtNumber(properties.area_km2, 3) + " км²";
    return '<div class="map-tooltip">' +
      '<div class="map-tooltip-title">Квартал ' + esc(properties.id) + '</div>' +
      '<div class="map-tooltip-row">Площадь: <b>' + esc(area) + '</b></div>' +
      '<div class="map-tooltip-row">' + esc(cfg.label) + ': <b>' + esc(valueLabel(properties, selected)) + '</b></div>' +
      '<div class="map-tooltip-note">' + esc(cfg.short || cfg.high) + '</div>' +
      '</div>';
  }

  function popupHtml(properties) {
    var cfg = metrics[selected];
    var area = fmtNumber(properties.area_km2, 3) + " км² (" + fmtNumber(properties.area_ha, 1) + " га)";
    var rows = [
      ["Площадь", area],
      ["Население", fmtNumber(properties.population, 0) + " чел."],
      ["Плотность", fmtNumber(properties.population_density, 0) + " чел./км²"],
      ["Зданий", fmtNumber(properties.building_count, 0) + " шт."],
      ["Школ", fmtNumber(properties.school_count, 0) + " шт."],
      ["Ёмкость школ", fmtNumber(properties.capacity, 0) + " мест"],
      ["Обеспеченность в пределах 15 минут", fmtNumber(properties.provision_strong, 3)],
      ["Обеспеченность с учётом дальних школ", fmtNumber(properties.provision_weak, 3)],
      ["Спрос", fmtNumber(properties.demand, 0) + " мест"],
      ["Дефицит", fmtNumber(properties.demand_left, 0) + " мест"],
      ["Остаток мест", fmtNumber(properties.capacity_left, 0) + " мест"],
      ["Доступность", fmtNumber(properties.area_accessibility, 1) + " мин"],
      ["FSI", fmtNumber(properties.fsi, 3)],
      ["GSI", fmtNumber(properties.gsi, 3)],
      ["MXI", fmtNumber(properties.mxi, 3)],
      ["L (этажность)", fmtNumber(properties.l, 1)],
      ["OSR", fmtNumber(properties.osr, 2)],
      ["Землепользование", categoryLabel("land_use", properties.land_use)],
      ["Spacematrix", categoryLabel("morphotype_spacematrix", properties.morphotype_spacematrix)],
      ["Strelka", categoryLabel("morphotype_strelka", properties.morphotype_strelka)],
      ["Кластер Spacematrix", categoryLabel("spacematrix_cluster_label", properties.spacematrix_cluster_label)]
    ];
    var html = '<div class="map-popup">' +
      '<div class="map-popup-kicker">Квартал</div>' +
      '<div class="map-popup-title">Квартал ' + esc(properties.id) + '</div>' +
      '<div class="map-popup-current">' +
        '<div class="map-popup-current-label">' + esc(cfg.label) + '</div>' +
        '<div class="map-popup-current-value">' + esc(valueLabel(properties, selected)) + '</div>' +
      '</div>' +
      '<div class="map-popup-description">' + esc(cfg.description) + '</div>' +
      '<table class="map-popup-table"><tbody>';
    rows.forEach(function (row) {
      html += cell(row[0], row[1]);
    });
    html += '</tbody></table></div>';
    return html;
  }

  function styleFor(feature) {
    var properties = feature && feature.properties ? feature.properties : {};
    var cfg = metrics[selected];
    var value = valueFor(properties, cfg);
    var isMissing = value === null;
    var isZero = !isMissing && cfg.type !== "categorical" && cfg.zero_separate && Number(value) === 0;
    var fill = colorFor(value, cfg);
    return {
      color: objectMeta.outline_color,
      weight: outlinesOn ? 0.3 : 0,
      opacity: outlinesOn ? 1 : 0,
      fillColor: fill,
      fillOpacity: isMissing ? 1 : (isZero ? 1 : 0.9)
    };
  }

  function bindBlock(feature, layer) {
    layer.bindPopup(function () {
      return popupHtml(layer.feature.properties);
    }, {maxWidth: 370, minWidth: 260, className: "tyumen-popup"});
    layer.bindTooltip(function () {
      return tooltipHtml(layer.feature.properties);
    }, {sticky: true, direction: "auto", className: "tyumen-tooltip", opacity: 1});
    layer.on("popupopen", function () { activePopupLayer = layer; });
    layer.on("popupclose", function () {
      if (activePopupLayer === layer) activePopupLayer = null;
    });
    layer.on("tooltipopen", function () { activeTooltipLayer = layer; });
    layer.on("tooltipclose", function () {
      if (activeTooltipLayer === layer) activeTooltipLayer = null;
    });
  }

  // Вода: на обзорном масштабе — только крупные объекты, при приближении — все.
  var WATER_ZOOM = 13;
  var WATER_LARGE_AREA = 0.05;
  var WATER_LARGE_LENGTH = 1.0;

  function waterStyle(feature) {
    var geometry = feature && feature.geometry ? feature.geometry : null;
    var type = geometry ? geometry.type : "";
    var properties = feature && feature.properties ? feature.properties : {};
    var area = Number(properties.area_km2 || 0);
    var length = Number(properties.length_km || 0);
    var isLine = type === "LineString" || type === "MultiLineString";
    var isLarge = isLine ? length >= WATER_LARGE_LENGTH : area >= WATER_LARGE_AREA;
    if (!isLarge && map.getZoom() < WATER_ZOOM) {
      return { weight: 0, opacity: 0, fill: false, fillOpacity: 0 };
    }
    if (isLine) {
      return {
        color: objectMeta.water_color,
        weight: 0.8,
        opacity: 0.8,
        fill: false
      };
    }
    return {
      color: objectMeta.water_color,
      weight: 0,
      opacity: 0.6,
      fill: true,
      fillColor: objectMeta.water_color,
      fillOpacity: 0.6
    };
  }

  var waterLayer = L.geoJSON(waterData, {
    pane: "waterPane",
    interactive: false,
    style: waterStyle
  });

  var thematicLayer = L.geoJSON(blockData, {
    pane: "blocksPane",
    style: styleFor,
    onEachFeature: bindBlock
  }).addTo(map);

  var schoolLayer = L.geoJSON(schoolData, {
    pane: "schoolsPane",
    pointToLayer: function (feature, latlng) {
      return L.circleMarker(latlng, {
        pane: "schoolsPane",
        radius: 2.3,
        color: "#2E2E2E",
        weight: 0.6,
        opacity: 0.8,
        fill: true,
        fillColor: "#2E2E2E",
        fillOpacity: 0.8
      });
    },
    onEachFeature: function (feature, layer) {
      var p = feature.properties;
      var source = p.capacity_source === "modeled_250" ? "модельная, 250 мест" : p.capacity_source;
      layer.bindPopup(
        '<div class="map-popup"><div class="map-popup-kicker">Школа</div>' +
        '<div class="map-popup-title">' + esc(p.name) + '</div>' +
        '<table class="map-popup-table"><tbody>' +
        cell("Ёмкость", fmtNumber(p.capacity, 0) + " мест") +
        cell("Источник ёмкости", source) +
        '</tbody></table></div>',
        {maxWidth: 320}
      );
      layer.bindTooltip(esc(p.name), {sticky: true, direction: "top"});
    }
  });

  var boundaryLayer = L.geoJSON(boundaryData, {
    pane: "boundaryPane",
    interactive: false,
    style: function () {
      return {
        color: "#262626",
        weight: 1.6,
        opacity: 1,
        fill: false
      };
    }
  }).addTo(map);

  L.control.attribution({
    position: "bottomright",
    prefix: '<a href="https://leafletjs.com/" target="_blank" rel="noopener noreferrer">Leaflet</a>'
  }).addTo(map).addAttribution(
    'Данные: © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">' +
    'OpenStreetMap contributors</a> (ODbL 1.0) · расчёты: BlocksNet 1.0.0a9 · слои: локальные данные исследования'
  );

  function renderLegend() {
    var cfg = metrics[selected];
    document.getElementById("metric-label").textContent = cfg.label;
    document.getElementById("metric-unit").textContent = cfg.unit ? ("Единица: " + cfg.unit) : "Категориальная шкала";
    document.getElementById("metric-description").textContent = cfg.short || cfg.description;

    var numericLegend = document.getElementById("numeric-legend");
    var categoryLegend = document.getElementById("category-legend");
    var scale = document.getElementById("legend-scale");
    var ticks = document.getElementById("legend-ticks");
    var special = document.getElementById("legend-special");
    var classBox = document.getElementById("legend-classes");
    var meta = document.getElementById("metric-meta");

    scale.innerHTML = "";
    ticks.innerHTML = "";
    special.innerHTML = "";
    categoryLegend.innerHTML = "";
    if (classBox) classBox.innerHTML = "";

    if (cfg.type === "categorical") {
      numericLegend.style.display = "none";
      categoryLegend.style.display = "block";
      cfg.categories.forEach(function (category) {
        var row = document.createElement("div");
        row.className = "category-row";
        row.innerHTML = '<span class="swatch" style="background:' + category.color + '"></span>' +
          '<span class="category-name">' + esc(category.label) + '</span>' +
          '<span class="category-count">' + fmtNumber(category.count, 0) + '</span>';
        categoryLegend.appendChild(row);
      });
      if (cfg.missing_count) {
        var missing = document.createElement("div");
        missing.className = "category-row special-row";
        missing.innerHTML = '<span class="swatch" style="background:' + cfg.missing_color + '"></span>' +
          '<span class="category-name">Нет данных</span>' +
          '<span class="category-count">' + fmtNumber(cfg.missing_count, 0) + '</span>';
        categoryLegend.appendChild(missing);
      }
      meta.textContent = "Категорий: " + cfg.categories.length + " · кварталов: " + fmtNumber(cfg.n, 0) +
        (cfg.missing_count ? " · нет данных: " + fmtNumber(cfg.missing_count, 0) : "");
      return;
    }

    numericLegend.style.display = "block";
    categoryLegend.style.display = "none";

    if (cfg.type === "constant") {
      scale.innerHTML = '<span class="constant-swatch" style="background:' + (cfg.colors[0] || "#bdbdbd") + '"></span>';
      ticks.innerHTML = '<span class="constant-label">' + (cfg.constant_value === null ? "нет данных" : fmtNumber(cfg.constant_value, cfg.decimals)) + '</span>';
      meta.textContent = cfg.n ? ("Все значения одинаковы · кварталов: " + fmtNumber(cfg.n, 0)) : "Нет сохранённых значений";
      return;
    }

    cfg.colors.forEach(function (color) {
      var segment = document.createElement("span");
      segment.style.background = color;
      scale.appendChild(segment);
    });
    cfg.breaks.forEach(function (value) {
      var label = document.createElement("span");
      label.textContent = fmtNumber(value, cfg.decimals);
      ticks.appendChild(label);
    });

    var counts = classCounts(cfg);
    if (classBox) {
      cfg.colors.forEach(function (color, index) {
        var low = cfg.breaks[index];
        var high = cfg.breaks[index + 1];
        var rangeText = (high === undefined || high === null)
          ? (fmtNumber(low, cfg.decimals) + " и выше")
          : (fmtNumber(low, cfg.decimals) + " – " + fmtNumber(high, cfg.decimals));
        var row = document.createElement("div");
        row.className = "legend-class";
        row.innerHTML = '<span class="swatch" style="background:' + color + '"></span>' +
          '<span class="legend-class-range">' + esc(rangeText) + '</span>' +
          '<span class="legend-class-count">' + fmtNumber(counts[index] || 0, 0) + ' кв.</span>';
        classBox.appendChild(row);
      });
    }

    if (cfg.zero_separate) {
      var zeroRow = document.createElement("div");
      zeroRow.className = "special-row";
      zeroRow.innerHTML = '<span class="swatch" style="background:' + cfg.zero_color + '"></span>' +
        '<span class="special-label">' + esc(cfg.zero_label) + '</span>' +
        '<span class="special-count">' + fmtNumber(cfg.zero_count, 0) + ' кв.</span>';
      special.appendChild(zeroRow);
    }
    if (cfg.missing_count) {
      var missingRow = document.createElement("div");
      missingRow.className = "special-row";
      missingRow.innerHTML = '<span class="swatch" style="background:' + cfg.missing_color + '"></span>' +
        '<span class="special-label">Нет данных / показатель не применим</span>' +
        '<span class="special-count">' + fmtNumber(cfg.missing_count, 0) + ' кв.</span>';
      special.appendChild(missingRow);
    }

    var range = "Диапазон: " + fmtNumber(cfg.data_min, cfg.decimals) + " — " + fmtNumber(cfg.data_max, cfg.decimals);
    meta.textContent = range + " · n=" + fmtNumber(cfg.n, 0) +
      (cfg.missing_count ? " · нет данных: " + fmtNumber(cfg.missing_count, 0) : "") +
      (cfg.zero_count ? " · нулей: " + fmtNumber(cfg.zero_count, 0) : "");
  }

  function renderChart() {
    var cfg = metrics[selected];
    var chart = metricCharts[selected] || {};
    var image = document.getElementById("metric-chart-image");
    var caption = document.getElementById("metric-chart-caption");
    var title = document.getElementById("metric-chart-title");
    if (image && chart.src) {
      image.src = chart.src;
      image.alt = "Распределение показателя: " + (cfg ? cfg.label : "");
    }
    if (caption) caption.textContent = chart.caption || "";
    if (title) title.textContent = "Распределение: " + (cfg ? cfg.label : "");
  }

  function applyMetric(key) {
    if (!metrics[key]) return;
    selected = key;
    thematicLayer.setStyle(styleFor);
    renderLegend();
    renderChart();
    syncToolbar();
    if (activePopupLayer && activePopupLayer.isPopupOpen() && activePopupLayer.feature) {
      activePopupLayer.setPopupContent(popupHtml(activePopupLayer.feature.properties));
    }
    if (activeTooltipLayer && activeTooltipLayer.isTooltipOpen() && activeTooltipLayer.feature) {
      activeTooltipLayer.setTooltipContent(tooltipHtml(activeTooltipLayer.feature.properties));
    }
  }

  var primaryMetrics = __PRIMARY_METRICS__;

  function buildToolbar() {
    var buttonsBox = document.getElementById("metric-buttons");
    if (buttonsBox) {
      Object.keys(primaryMetrics).forEach(function (key) {
        if (!metrics[key]) return;
        var button = document.createElement("button");
        button.type = "button";
        button.className = "metric-button";
        button.setAttribute("data-metric", key);
        button.setAttribute("aria-pressed", "false");
        button.textContent = primaryMetrics[key];
        button.addEventListener("click", function () { applyMetric(key); });
        buttonsBox.appendChild(button);
      });
    }

    var select = document.getElementById("metric-select");
    if (!select) return;
    var groups = {};
    Object.keys(metrics).forEach(function (key) {
      var groupName = metrics[key].group || "Показатели";
      if (!groups[groupName]) groups[groupName] = [];
      groups[groupName].push(key);
    });
    Object.keys(groups).forEach(function (groupName) {
      var group = document.createElement("optgroup");
      group.label = groupName;
      groups[groupName].forEach(function (key) {
        var option = document.createElement("option");
        option.value = key;
        option.textContent = metrics[key].label;
        group.appendChild(option);
      });
      select.appendChild(group);
    });
    select.addEventListener("change", function () {
      applyMetric(select.value);
    });
  }

  function syncToolbar() {
    var cfg = metrics[selected];
    var select = document.getElementById("metric-select");
    if (select && select.value !== selected) select.value = selected;
    var name = document.getElementById("toolbar-metric-name");
    if (name) name.textContent = cfg ? cfg.label : "";
    var buttons = document.querySelectorAll("#metric-buttons .metric-button");
    for (var i = 0; i < buttons.length; i += 1) {
      var active = buttons[i].getAttribute("data-metric") === selected;
      buttons[i].className = active ? "metric-button is-active" : "metric-button";
      buttons[i].setAttribute("aria-pressed", active ? "true" : "false");
    }
  }

  function bindLayerToggles() {
    var waterToggle = document.getElementById("toggle-water");
    var schoolsToggle = document.getElementById("toggle-schools");
    var boundaryToggle = document.getElementById("toggle-boundary");
    var outlinesToggle = document.getElementById("toggle-outlines");
    if (waterToggle) {
      waterToggle.addEventListener("change", function (event) {
        if (event.target.checked) {
          map.addLayer(waterLayer);
          waterLayer.setStyle(waterStyle);
        } else {
          map.removeLayer(waterLayer);
        }
      });
    }
    if (schoolsToggle) {
      schoolsToggle.addEventListener("change", function (event) {
        if (event.target.checked) map.addLayer(schoolLayer); else map.removeLayer(schoolLayer);
      });
    }
    if (boundaryToggle) {
      boundaryToggle.addEventListener("change", function (event) {
        if (event.target.checked) map.addLayer(boundaryLayer); else map.removeLayer(boundaryLayer);
      });
    }
    if (outlinesToggle) {
      outlinesToggle.addEventListener("change", function (event) {
        outlinesOn = event.target.checked;
        thematicLayer.setStyle(styleFor);
      });
    }
  }

  map.on("zoomend", function () {
    if (map.hasLayer(waterLayer)) waterLayer.setStyle(waterStyle);
  });

  buildToolbar();
  bindLayerToggles();
  applyMetric(selected);

  var badge = document.createElement("div");
  badge.className = "map-badge";
  badge.textContent = "Тюмень · " + fmtNumber(objectMeta.blocks, 0) + " кварталов";
  map.getContainer().appendChild(badge);

  map.fitBounds(boundaryBounds, {padding: [24, 24], maxZoom: 13});
  thematicLayer.setStyle(styleFor);

  window.__tyumenDashboard = {
    selectMetric: applyMetric,
    getState: function () {
      return {
        selected: selected,
        metricLabel: metrics[selected].label,
        metricType: metrics[selected].type,
        legendUnit: document.getElementById("metric-unit").textContent,
        legendTicks: Array.prototype.map.call(document.querySelectorAll("#legend-ticks span"), function (node) { return node.textContent; }),
        legendClasses: Array.prototype.map.call(document.querySelectorAll("#legend-classes .legend-class"), function (node) { return node.textContent.replace(/\s+/g, " ").trim(); }),
        toolbarButtons: document.querySelectorAll("#metric-buttons .metric-button").length,
        activeToolbarButton: (document.querySelector("#metric-buttons .metric-button.is-active") || {}).textContent || "",
        metricCount: thematicLayer.getLayers().length,
        waterVisible: map.hasLayer(waterLayer),
        schoolsVisible: map.hasLayer(schoolLayer),
        boundaryVisible: map.hasLayer(boundaryLayer),
        outlinesVisible: outlinesOn,
        hasUkrainianFlag: !!document.querySelector(".leaflet-attribution-flag"),
        attributionText: (document.querySelector(".leaflet-control-attribution") || {}).textContent || ""
      };
    },
    map: map,
    blocksLayer: thematicLayer,
    metrics: metrics,
    objectMeta: objectMeta
  };

  window.setTimeout(function () {
    map.invalidateSize();
    map.fitBounds(boundaryBounds, {padding: [24, 24], maxZoom: 13});
  }, 80);
})();
</script>
"""


def make_map_script() -> str:
    script = MAP_SCRIPT_TEMPLATE
    replacements = {
        "__METRICS__": js_json(metric_config),
        "__PRIMARY_METRICS__": js_json(PRIMARY_METRICS),
        "__METRIC_CHARTS__": js_json(metric_charts),
        "__BLOCK_DATA__": js_json(block_data),
        "__WATER_DATA__": js_json(water_data),
        "__BOUNDARY_DATA__": js_json(boundary_data),
        "__SCHOOL_DATA__": js_json(school_data),
        "__OBJECT_META__": js_json(object_meta),
        "__BOUNDARY_BOUNDS__": js_json(boundary_bounds),
    }
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)
    return script


map_script = make_map_script()



# ---------------------------------------------------------------------------
# Выводы, карточки и пояснения
# ---------------------------------------------------------------------------
conclusions = [
    (
        "Итоговая структура кварталов: "
        f"{fmt(block_count, 0)} кварталов, площадь {fmt(area_total, 2)} км². "
        f"Максимальная площадь квартала — {fmt(max_area, 2)} км²; "
        f"кварталов больше 10 км² — {fmt(big_blocks, 0)}. "
        f"Застроенных кварталов (FSI > 0) — {fmt(developed_blocks, 0)}."
    ),
    (
        f"В кварталах города проживает {fmt(population, 0)} человек. "
        f"На топ-10 кварталов по населению приходится {fmt(top10_population_share, 1)}% населения, "
        "то есть распределение остаётся неравномерным."
    ),
    (
        f"В модели {fmt(school_count, 0)} школ с суммарной модельной вместимостью {fmt(capacity, 0)} мест. "
        f"Все {fmt(schools_modeled, 0)} школ получили оценку 250 мест, школ с вместимостью из OpenStreetMap — {fmt(schools_osm, 0)}: "
        "это расчётное допущение, а не подтверждённая вместимость школ города. "
        f"Средняя обеспеченность в пределах 15 минут — {fmt(mean_provision, 3)}. "
        f"Среди {fmt(int(populated.sum()), 0)} населённых кварталов у {fmt(zero_provision_count, 0)} обеспеченность равна нулю; "
        f"ещё у {fmt(missing_provision_count, 0)} показатель не применим, потому что спрос равен нулю."
    ),
    (
        f"Нормативная потребность в школьных местах — {fmt(demand_total, 0)} мест, "
        f"покрыто в пределах 15 минут — {fmt(demand_within_total, 0)} мест, "
        f"дефицит — {fmt(demand_left_total, 0)} мест. "
        "Это следует из сохранённых результатов расчёта обеспеченности, а не из нового расчёта."
    ),
    (
        f"Средняя пешеходная доступность территории — {fmt(mean_accessibility, 1)} мин, "
        f"медиана — {fmt(ref_results['area_accessibility'].median(), 1)} мин, "
        f"диапазон — {fmt(ref_results['area_accessibility'].min(), 1)}–{fmt(ref_results['area_accessibility'].max(), 1)} мин. "
        "Большие значения означают меньшую связность квартала с городом."
    ),
    (
        f"В землепользовании преобладает категория «{top_land_use_text}», "
        f"а среди морфотипов Spacematrix — «{top_morphotype_text}». "
        "У 5 252 кварталов преобладающая категория землепользования не определена; на карте это отдельная категория «нет данных»."
    ),
    (
        "MXI в BlocksNet 1.0.0a9 — это доля жилой поэтажной площади (жилая площадь делится на суммарную поэтажную), "
        "а не энтропийный индекс смешанности. Поэтому высокий MXI означает преимущественно жилую застройку, "
        "а не максимальное разнообразие функций."
    ),
    (
        f"Проверки качества: {'все пройдены' if checks_ok else 'есть замечания'}. "
        f"Покрытие суши — {fmt(coverage_land, 2)}%, пересечения кварталов — "
        f"{fmt(blocks_quality.get('overlap_area_km2'), 6)} км², площадь вне границы — "
        f"{fmt(blocks_quality.get('outside_area_km2'), 6)} км². "
        f"Максимальное расхождение двух показателей обеспеченности — {fmt(provision_diff_max, 6)}."
    ),
]
conclusions_html = "\n".join(f"<li>{html.escape(item)}</li>" for item in conclusions)

limitations = [
    "Исследование построено на уточнённой квартальной структуре: крупные кварталы разделены, геометрия и покрытие проверены. Сравнение с первоначальной нарезкой в дашборд не включено.",
    "Население — модельная оценка по жилой площади зданий, а не официальная статистика.",
    "Вместимость всех 70 школ после удаления дубликатов принята равной 250 местам: атрибута вместимости в OpenStreetMap нет. Суммарные 17 500 мест — расчётная величина по этому допущению, а не подтверждённая вместимость школ Тюмени.",
    "Оба показателя обеспеченности в этом расчёте совпадают, потому что школ дальше нормативных 15 минут не использовалось: ёмкость и спрос вне норматива равны нулю.",
    "Обеспеченность нормирована по нормативу 120 мест на 1000 жителей и по радиусу доступности 15 минут пешего пути. Это методика расчёта, а не административная норма и не оценка качества школьной сети.",
    f"Обеспеченность применима только к кварталам с населением: без населения нормативная потребность равна нулю, поэтому в {fmt(missing_provision_count, 0)} кварталах показатель не определён и остаётся белым на карте. Городская оценка обеспеченности относится к застроенной части территории.",
    "Время доступности — пешеходное, в минутах, по пешеходному графу; интермодальные варианты и общественный транспорт не учитываются.",
    "Доступность территории — расчёт по матрице пешеходного графа; матрица 12 805 × 12 805 не встраивается в страницу из-за размера (CSV около 1,09 ГБ).",
    "Геометрия кварталов на карте упрощена для отображения; таблицы, метрики и исходные файлы не пересчитывались.",
    "Числовые шкалы карты используют квантильные границы по фактическим значениям, а для долей (обеспеченность, MXI) — округлённые интервалы; это классификация для цвета, а не изменение данных. Выбросы не удаляются, а попадают в крайний интервал.",
    "MXI трактуется по фактической реализации BlocksNet 1.0.0a9 как доля жилой площади; это важно не путать с индексом смешанности функций.",
    "На карте нет сетевых тайлов. Leaflet CSS и JS подключаются с CDN jsDelivr; все геоданные и диаграммы встроены в страницу.",
    "Показатели относительной доступности, связности, центральности, вакантных территорий и услуг кроме школ в сохранённых данных отсутствуют и поэтому не выдумываются.",
]
limitations_html = "\n".join(f"<li>{html.escape(item)}</li>" for item in limitations)

explanation_cards = []
current_group = None
for key, cfg in metric_config.items():
    if cfg["group"] != current_group:
        current_group = cfg["group"]
        explanation_cards.append(
            f'<div class="explain-group"><h3>{html.escape(current_group)}</h3></div>'
        )
    explanation_cards.append(
        '<div class="explain-card">'
        f"<h4>{html.escape(cfg['label'])}</h4>"
        f"<p>{html.escape(cfg['description'])}</p>"
        f'<div class="explain-meta"><b>Единица:</b> {html.escape(cfg["unit"])}<br>'
        f"<b>Высокое значение:</b> {html.escape(cfg['high'])}<br>"
        f"<b>Низкое значение:</b> {html.escape(cfg['low'])}</div>"
        "</div>"
    )
explanations_html = "\n".join(explanation_cards)

cards = [
    ("Кварталы", fmt(block_count, 0), "проверенная квартальная структура"),
    ("Население", fmt(population, 0), "чел., модельная оценка"),
    ("Школы", fmt(school_count, 0), "70 после удаления дубликатов; 82 объекта в исходных данных"),
    ("Модельная вместимость школ", fmt(capacity, 0), "мест; 250 мест на школу — допущение"),
    ("Средняя обеспеченность", fmt(mean_provision, 3), "в пределах 15 минут, среди кварталов с населением"),
    ("Средняя доступность", f"{fmt(mean_accessibility, 1)} мин", "пешеходная, с учётом площади"),
    ("Спрос на места", fmt(demand_total, 0), "мест по нормативу 120 на 1000 жителей"),
]
cards_html = "\n".join(
    f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
    f'<div class="kpi-value">{html.escape(value)}</div>'
    f'<div class="kpi-note">{html.escape(note)}</div></div>'
    for label, value, note in cards
)

generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")
metric_count = len(metric_config)

PAGE_TEMPLATE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Кварталы и метрики города: исследование Тюмени</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css">
<style>
:root {
  --bg: #ffffff;
  --surface: #ffffff;
  --card: #f5f5f5;
  --ink: #171717;
  --text: #262626;
  --muted: #737373;
  --line: #e5e5e5;
  --line-strong: #d4d4d4;
  --accent: #171717;
  --map-height: 790px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  line-height: 1.5;
}
.wrap { max-width: 1520px; margin: 0 auto; padding: 28px 22px 52px; }
header.hero {
  background: var(--surface);
  border: 1px solid var(--line);
  border-top: 4px solid var(--accent);
  border-radius: 12px;
  padding: 28px 30px;
  margin-bottom: 24px;
}
.eyebrow { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 8px; }
header h1 { margin: 0 0 10px; font-size: 31px; line-height: 1.2; color: var(--ink); }
header p { margin: 0; max-width: 1040px; color: var(--muted); }
.meta { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
.meta span { border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 12px; padding: 4px 10px; }
h2 { margin: 32px 0 12px; font-size: 22px; color: var(--ink); }
h3 { margin: 0 0 10px; font-size: 15px; color: var(--ink); }
.section-note { color: var(--muted); margin: -4px 0 14px; font-size: 13px; max-width: 1120px; }
.method {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 18px 20px;
  color: var(--text);
}
.method p { margin: 0 0 10px; }
.method p:last-child { margin-bottom: 0; }
.method ul { margin: 8px 0 0 20px; padding: 0; }
.method li { margin: 6px 0; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 12px; margin: 18px 0 8px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 15px 16px; }
.kpi-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.kpi-value { font-size: 24px; font-weight: 700; margin: 5px 0 2px; color: var(--ink); }
.kpi-note { color: var(--muted); font-size: 12px; }
.analytics-layout {
  display: grid;
  grid-template-columns: minmax(0, 7fr) minmax(320px, 3fr);
  gap: 16px;
  align-items: start;
}
.analytics-main { min-width: 0; }
.analytics-side { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.panel-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 13px 14px 14px;
}
.panel-card-title {
  color: var(--ink);
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 8px;
  letter-spacing: .01em;
}
.chart-panel img {
  width: 100%;
  height: auto;
  display: block;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #ffffff;
}
.chart-caption { color: var(--muted); font-size: 11.5px; margin-top: 8px; }
.map-toolbar {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 14px 13px;
  margin-bottom: 12px;
}
.toolbar-head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 9px;
}
.toolbar-kicker {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .09em;
}
.toolbar-value { color: var(--ink); font-size: 15px; font-weight: 700; }
.metric-buttons { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 10px; }
.metric-button {
  background: #ffffff;
  color: #525252;
  border: 1px solid #d4d4d4;
  border-radius: 999px;
  padding: 6px 13px;
  font: 12.5px/1.3 -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  cursor: pointer;
}
.metric-button:hover { background: #f5f5f5; color: #171717; }
.metric-button.is-active { background: #171717; color: #ffffff; border-color: #171717; }
.metric-button:focus-visible { outline: 2px solid #171717; outline-offset: 1px; }
.toolbar-select-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.toolbar-select-label { color: var(--muted); font-size: 11.5px; }
.toolbar-select-row select {
  flex: 1 1 220px;
  min-width: 180px;
  background: #ffffff;
  color: #262626;
  border: 1px solid #d4d4d4;
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12.5px;
}
.map-frame {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 8px;
  overflow: hidden;
}
.tyumen-map {
  width: 100%;
  height: var(--map-height);
  border-radius: 8px;
  background: #f5f5f5;
  position: relative;
  z-index: 1;
}
.map-fallback { padding: 40px 24px; color: #262626; font-size: 15px; max-width: 760px; }
.map-badge {
  position: absolute;
  bottom: 12px;
  left: 12px;
  z-index: 500;
  pointer-events: none;
  background: rgba(255,255,255,.9);
  border: 1px solid var(--line-strong);
  border-radius: 6px;
  padding: 6px 9px;
  color: #525252;
  font-size: 11px;
  letter-spacing: .02em;
}
/* Leaflet: монохромные контролы и панель показателя */
.leaflet-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }
.leaflet-control-zoom a {
  background: #ffffff !important;
  color: #262626 !important;
  border-color: #d4d4d4 !important;
}
.leaflet-control-zoom a:hover { background: #f5f5f5 !important; }
.leaflet-control-attribution {
  background: rgba(255,255,255,.93) !important;
  color: #737373 !important;
  border: 1px solid #e5e5e5 !important;
  border-radius: 4px !important;
  padding: 2px 6px !important;
  font-size: 10px !important;
}
.leaflet-control-attribution a { color: #525252 !important; }
.leaflet-control-attribution .leaflet-attribution-flag { display: none !important; }
.panel-divider { border-top: 1px solid #e5e5e5; margin: 10px 0; }
.legend-title { color: #525252; font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
.metric-label { color: var(--ink); font-size: 13px; font-weight: 700; margin-bottom: 1px; }
.metric-unit { color: var(--muted); font-size: 10.5px; margin-bottom: 6px; }
.legend-scale { display: flex; height: 14px; border: 1px solid #d4d4d4; border-radius: 2px; overflow: hidden; }
.legend-scale span { flex: 1 1 0; min-width: 0; }
.legend-ticks { display: flex; justify-content: space-between; gap: 2px; color: #737373; font-size: 9.5px; margin-top: 3px; }
.legend-ticks span { white-space: nowrap; }
.legend-classes { margin-top: 7px; }
.legend-class { display: flex; align-items: center; gap: 7px; color: #525252; font-size: 11px; margin: 3px 0; }
.legend-class-range { flex: 1 1 auto; font-variant-numeric: tabular-nums; }
.legend-class-count { color: #737373; font-variant-numeric: tabular-nums; white-space: nowrap; }
.legend-special { margin-top: 6px; }
.special-row, .category-row { display: flex; align-items: center; gap: 6px; color: #525252; font-size: 10.5px; margin: 4px 0; }
.special-label { flex: 1 1 auto; }
.special-count { color: #737373; font-variant-numeric: tabular-nums; white-space: nowrap; }
.swatch { display: inline-block; flex: 0 0 15px; width: 15px; height: 10px; border-radius: 2px; border: 1px solid #d4d4d4; }
.swatch-water { background: #707070; border-color: #707070; }
.swatch-school { width: 9px; height: 9px; flex-basis: 9px; border-radius: 50%; background: #2e2e2e; border-color: #2e2e2e; }
.swatch-boundary { height: 0; border: 0; border-top: 2px solid #262626; background: transparent; }
.swatch-outline { height: 0; border: 0; border-top: 1px solid #8a8a8a; background: transparent; }
.category-list { display: none; max-height: 205px; overflow-y: auto; padding-right: 2px; }
.category-name { flex: 1; }
.category-count { color: #737373; font-variant-numeric: tabular-nums; }
.metric-description { color: #737373; font-size: 10.7px; margin-top: 7px; }
.metric-highlow { color: #525252; font-size: 10.5px; margin-top: 6px; }
.metric-highlow b { color: #525252; }
.metric-meta { color: #a3a3a3; font-size: 10px; margin-top: 5px; }
.constant-swatch { display: block; height: 12px; border-radius: 2px; }
.constant-label { display: block; color: #525252; font-size: 10.5px; margin-top: 4px; }
.layer-row { display: flex; align-items: center; gap: 7px; color: #525252; font-size: 11.5px; margin: 5px 0; cursor: pointer; }
.layer-row input { accent-color: #525252; margin: 0; flex: 0 0 auto; }
.layer-row span:last-child { flex: 1 1 auto; min-width: 0; }
.layer-note { color: #a3a3a3; font-size: 10px; margin-top: 6px; }
.leaflet-popup-content-wrapper, .leaflet-tooltip {
  background: #ffffff;
  color: #262626;
  border: 1px solid #d4d4d4;
  box-shadow: 0 4px 18px rgba(23,23,23,.16);
}
.leaflet-popup-tip { background: #ffffff; }
.tyumen-tooltip { padding: 0 !important; border-radius: 6px !important; }
.map-tooltip { padding: 8px 10px; min-width: 190px; }
.map-tooltip-title { font-weight: 700; font-size: 12px; margin-bottom: 3px; }
.map-tooltip-row { font-size: 11.5px; color: #525252; }
.map-tooltip-note { border-top: 1px solid #e5e5e5; color: #737373; font-size: 10.5px; margin-top: 5px; padding-top: 4px; max-width: 260px; }
.map-popup { width: 330px; max-width: 78vw; padding: 2px 1px; }
.map-popup-kicker { color: #737373; font-size: 9.5px; text-transform: uppercase; letter-spacing: .08em; }
.map-popup-title { color: #171717; font-size: 16px; font-weight: 700; margin: 1px 0 8px; }
.map-popup-current { background: #f5f5f5; border: 1px solid #e5e5e5; border-radius: 6px; padding: 8px 9px; }
.map-popup-current-label { color: #525252; font-size: 10.5px; }
.map-popup-current-value { color: #171717; font-size: 16px; font-weight: 700; }
.map-popup-description { color: #737373; font-size: 10.5px; margin: 7px 0; }
.map-popup-table { border-collapse: collapse; width: 100%; font-size: 10.8px; }
.map-popup-table th { text-align: left; color: #737373; font-weight: 500; padding: 3px 7px 3px 0; vertical-align: top; }
.map-popup-table td { color: #262626; padding: 3px 0; text-align: right; vertical-align: top; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 16px; }
.chart-card { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
.chart-card img { width: 100%; height: auto; display: block; }
.chart-card figcaption { color: var(--muted); font-size: 12px; margin-top: 8px; }
.explanations { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; margin-top: 14px; }
.method-details {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
}
.method-details summary {
  cursor: pointer;
  color: var(--ink);
  font-size: 14px;
  font-weight: 700;
  list-style: revert;
}
.method-details summary::marker { color: #737373; }
.explain-group { grid-column: 1 / -1; margin-top: 6px; }
.explain-group h3 { color: var(--ink); font-size: 14px; font-weight: 700; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
.explain-card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 15px 16px; }
.explain-card h4 { margin: 0 0 6px; font-size: 13.5px; font-weight: 700; color: var(--ink); }
.explain-card p { margin: 0; color: var(--muted); font-size: 12.5px; }
.explain-meta { border-top: 1px solid var(--line-strong); color: #525252; font-size: 11.5px; margin-top: 10px; padding-top: 8px; }
.explain-meta b { color: #525252; }
.table-card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin: 14px 0; overflow-x: auto; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data-table th { background: var(--card); color: var(--ink); text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line-strong); position: sticky; top: 0; white-space: nowrap; }
table.data-table td { padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.data-table tr:nth-child(even) td { background: #fcfcfc; }
table.data-table tr:hover td { background: #f5f5f5; }
ul { margin: 10px 0 0 20px; padding: 0; }
li { margin: 7px 0; }
.two-col { display: grid; grid-template-columns: 1.15fr .85fr; gap: 18px; }
.two-col h3 { margin-top: 0; }
.warning-box { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; color: var(--text); font-size: 12.5px; margin: 12px 0; }
footer { color: var(--muted); font-size: 12px; margin-top: 30px; border-top: 1px solid var(--line); padding-top: 16px; }
footer p { margin: 5px 0; }
@media (max-width: 1240px) {
  .analytics-layout { grid-template-columns: 1fr; }
}
@media (max-width: 980px) {
  .two-col { grid-template-columns: 1fr; }
  .charts { grid-template-columns: 1fr; }
}
@media (max-width: 680px) {
  .wrap { padding: 18px 12px 36px; }
  header.hero { padding: 22px 18px; }
  header h1 { font-size: 25px; }
  :root { --map-height: 660px; }
  .map-popup { width: 280px; }
}
</style>
</head>
<body>
<div class="wrap">
<header class="hero">
  <div class="eyebrow">ДОМАШНЕЕ ЗАДАНИЕ № 1 · BLOCKSNET</div>
  <h1>Кварталы и метрики города: исследование Тюмени</h1>
  <p>Анализ квартальной структуры, застройки, населения, городской инфраструктуры и пространственной доступности Тюмени.</p>
  <div class="meta">
    <span>Территория: Тюмень</span>
    <span>Сформировано: __GENERATED_AT__</span>
    <span>Кварталов: __BLOCK_COUNT__</span>
    <span>Показателей в переключателе: __METRIC_COUNT__</span>
  </div>
</header>

<section>
  <h2>Методика исследования</h2>
  <div class="method">
    <p>Исследование выполнено на основе практического занятия № 2: кварталы и показатели рассчитаны в ноутбуке <code>examples/tyumen_pipeline_refined.ipynb</code>, а эта страница собирается отдельным скриптом <code>examples/tyumen_dashboard_refined.py</code> из уже готовых файлов. При сборке страницы данные не скачиваются, матрица доступности и показатели не пересчитываются.</p>
    <ul>
      <li><b>Исходные пространственные данные:</b> граница города, вода, школы, здания, дороги и функциональные зоны получены из OpenStreetMap через локальный архив PBF (Geofabrik, Уральский федеральный округ, лицензия ODbL 1.0). Кварталы построены инструментами BlocksNet 1.0.0a9.</li>
      <li><b>Уточнение квартальной структуры:</b> первоначальная нарезка была пересмотрена после проверки геометрии. Крупные кварталы площадью более 1 км² дополнительно разделены по жилой застройке: из 11 625 кварталов получилось 12 805. Пустые крупные полигоны без зданий искусственно не дробились. Итоговая структура проверена: пересечений кварталов нет, площадь вне границы города — 0 км², покрытие суши — 99,95 %, невалидных геометрий — 0.</li>
      <li><b>Модельные оценки, а не факты:</b> население кварталов восстановлено расчётным путём по жилой площади зданий — это модельная оценка, а не официальная статистика. Вместимость 250 мест принята для всех 70 школ, потому что в OpenStreetMap нет атрибута вместимости; суммарные 17 500 мест — расчётная величина по этому допущению, а не подтверждённая вместимость школ Тюмени.</li>
      <li><b>Расчётные показатели:</b> FSI, GSI, MXI, L, OSR, доли землепользования, морфотипы Spacematrix и Strelka, обеспеченность школами и время пешеходной доступности взяты из сохранённых результатов расчёта, а не введены вручную.</li>
      <li><b>Производные только для отображения:</b> площадь в гектарах, население на км², округление значений на карте и классификация цветом. Исходные файлы с результатами не изменяются.</li>
    </ul>
  </div>
</section>

<section>
  <h2>Ключевые показатели</h2>
  <div class="kpis">__CARDS__</div>
</section>

<section id="map">
  <h2>Интерактивная карта Тюмени</h2>
  <p class="section-note">Выберите показатель в переключателе над картой: окраска кварталов, легенда и график справа обновляются одновременно, центр и масштаб карты сохраняются. Нажмите на квартал, чтобы увидеть его номер, площадь и значение выбранного показателя.</p>
  <div class="analytics-layout">
    <div class="analytics-main">
      <div class="map-toolbar">
        <div class="toolbar-head">
          <span class="toolbar-kicker">Показатель карты</span>
          <span class="toolbar-value" id="toolbar-metric-name">—</span>
        </div>
        <div class="metric-buttons" id="metric-buttons" role="group" aria-label="Быстрый выбор показателя карты"></div>
        <div class="toolbar-select-row">
          <label class="toolbar-select-label" for="metric-select">Полный список показателей</label>
          <select id="metric-select" aria-label="Выбор показателя карты"></select>
        </div>
      </div>
      <div class="map-frame">
        <div id="tyumen-map" class="tyumen-map" role="application" aria-label="Интерактивная тематическая карта кварталов Тюмени"></div>
      </div>
    </div>
    <aside class="analytics-side" aria-label="Аналитика выбранного показателя">
      <div class="panel-card">
        <div class="panel-card-title">Условные обозначения</div>
        <div id="metric-label" class="metric-label"></div>
        <div id="metric-unit" class="metric-unit"></div>
        <div id="numeric-legend">
          <div id="legend-scale" class="legend-scale"></div>
          <div id="legend-ticks" class="legend-ticks"></div>
          <div id="legend-classes" class="legend-classes"></div>
          <div id="legend-special" class="legend-special"></div>
        </div>
        <div id="category-legend" class="category-list"></div>
        <div id="metric-description" class="metric-description"></div>
        <div id="metric-meta" class="metric-meta"></div>
      </div>
      <div class="panel-card">
        <div class="panel-card-title">Слои карты</div>
        <label class="layer-row"><input id="toggle-water" type="checkbox">
          <span class="swatch swatch-water"></span><span>Вода</span></label>
        <label class="layer-row"><input id="toggle-schools" type="checkbox">
          <span class="swatch swatch-school"></span><span>Школы</span></label>
        <label class="layer-row"><input id="toggle-boundary" type="checkbox" checked>
          <span class="swatch swatch-boundary"></span><span>Граница города</span></label>
        <label class="layer-row"><input id="toggle-outlines" type="checkbox" checked>
          <span class="swatch swatch-outline"></span><span>Кварталы</span></label>
      </div>
      <div class="panel-card chart-panel">
        <div class="panel-card-title" id="metric-chart-title">Распределение</div>
        <img id="metric-chart-image" alt="Распределение выбранного показателя по кварталам">
        <div id="metric-chart-caption" class="chart-caption"></div>
      </div>
    </aside>
  </div>
</section>

<section>
  <h2>Методика и определения</h2>
  <p class="section-note">Определения всех показателей, единицы измерения и краткая интерпретация высоких и низких значений. Раздел свёрнут по умолчанию.</p>
  <details class="method-details">
    <summary>Показатели и их определения (__METRIC_COUNT__)</summary>
    <div class="explanations">__EXPLANATIONS__</div>
  </details>
</section>

<section>
  <h2>Сводная статистика</h2>
  <p class="section-note">Сводка по всем кварталам города: количество наблюдений, среднее, стандартное отклонение, минимум, перцентили и максимум.</p>
  <div class="table-card">__SUMMARY_TABLE__</div>
</section>

<section>
  <h2>Все числовые показатели</h2>
  <p class="section-note">Среднее, медиана, минимум, максимум, число нулевых и число кварталов без данных по всем числовым показателям расчёта. Бесконечные значения исключены из статистики, но не изменены в исходных файлах.</p>
  <div class="table-card">__NUMERIC_TABLE__</div>
</section>

<section>
  <h2>Топ-10 кварталов по населению</h2>
  <div class="table-card">__TOP_POPULATION__</div>
</section>

<section>
  <h2>Топ-10 кварталов по времени доступности</h2>
  <p class="section-note">Наибольшее средневзвешенное время пешей доступности — потенциальные проблемные зоны связности.</p>
  <div class="table-card">__TOP_ACCESSIBILITY__</div>
</section>

<section>
  <h2>Населённые кварталы с минимальной обеспеченностью</h2>
  <p class="section-note">Только кварталы с населением и применимым показателем; сортировка по обеспеченности школами, затем по населению.</p>
  <div class="table-card">__TOP_PROVISION__</div>
</section>

<section>
  <h2>Школьная сеть и обеспеченность</h2>
  <p class="section-note">Оба показателя обеспеченности, нормативная потребность, покрытие в пределах 15 минут, дефицит, остаток мест и распределение спроса между школами. В этом расчёте обеспеченность с учётом дальних школ совпадает с обеспеченностью в пределах норматива, потому что школ дальше 15 минут не использовалось.</p>
  <div class="table-card">__PROVISION_TOTALS__</div>
  <div class="table-card">__PROVISION_TABLE__</div>
  <div class="table-card">__TOP_DEFICIT__</div>
  <div class="table-card">__LINKS_TABLE__</div>
</section>

<section>
  <h2>Школы: вместимость и источник данных</h2>
  <p class="section-note">70 школ после удаления дубликатов. У всех 70 вместимость модельная — 250 мест на школу, потому что атрибут вместимости в OpenStreetMap отсутствует. Это расчётное допущение, а не подтверждённая вместимость.</p>
  <div class="table-card">__SCHOOLS_TABLE__</div>
</section>

<section>
  <h2>Категориальные распределения</h2>
  <p class="section-note">Землепользование, морфотипы Spacematrix и Strelka, кластеры Spacematrix.</p>
  <div class="table-card">__CATEGORICAL_TABLE__</div>
</section>

<section>
  <h2>Портреты кластеров Spacematrix</h2>
  <p class="section-note">Характеристики групп кварталов, полученные методом Spacematrix по показателям этажности, FSI и MXI.</p>
  <div class="table-card">__CLUSTERS_TABLE__</div>
</section>

<section>
  <h2>Соответствие исходному заданию</h2>
  <p class="section-note">Проверка по требованиям практического занятия: каждый показатель либо показан в данных и на странице, либо явно отмечен как отсутствующий. Ничего не домысливается: для отсутствующих расчётов в таблице приведена причина.</p>
  <div class="table-card">__COVERAGE_TABLE__</div>
</section>

<section>
  <h2>Проверки качества данных</h2>
  <p class="section-note">Автоматические проверки геометрии кварталов, данных, матрицы доступности и полноты показателей.</p>
  <div class="table-card">__QUALITY_TABLE__</div>
  <div class="table-card">__QUALITY_META_TABLE__</div>
</section>

<section>
  <h2>Геометрия кварталов и матрица доступности</h2>
  <p class="section-note">Проверка геометрии после нарезки, параметры разделения крупных кварталов и состав матрицы времени в пути.</p>
  <div class="table-card">__BLOCKS_QUALITY_TABLE__</div>
  <div class="table-card">__BLOCKS_QUALITY_STATS_TABLE__</div>
  <div class="table-card">__SPLIT_META_TABLE__</div>
  <div class="table-card">__MATRIX_META_TABLE__</div>
</section>

<section>
  <h2>Выводы и ограничения</h2>
  <div class="two-col">
    <div>
      <h3>Выводы</h3>
      <ul>__CONCLUSIONS__</ul>
    </div>
    <div>
      <h3>Ограничения</h3>
      <ul>__LIMITATIONS__</ul>
    </div>
  </div>
</section>

<footer>
  <h2>Источники данных</h2>
  <p><b>Источники данных:</b> OpenStreetMap contributors (ODbL 1.0) — локальный архив PBF Уральского федерального округа; расчёты — BlocksNet 1.0.0a9; пешеходный граф (69 733 узла, 194 308 рёбер) и матрица времени в пути в минутах пешего передвищения рассчитаны для Тюмени; все показатели прочитаны из сохранённых файлов расчёта.</p>
  <p><b>Модельные допущения:</b> население восстановлено по жилой площади зданий; вместимость школ принята равной 250 местам; норматив потребности — 120 мест на 1000 жителей; норматив доступности — 15 минут пешего пути. Эти величины — оценки по методике расчёта, а не официальные данные.</p>
  <p><b>Карта:</b> Leaflet 1.9.3, локальные векторные слои без сетевых тайлов. <b>Внешние ресурсы:</b> только Leaflet CSS и JS с CDN jsDelivr; все геоданные, таблицы и диаграммы встроены в этот файл.</p>
  <div class="table-card">__SOURCE_META_TABLE__</div>
  <p>Скрипт сборки страницы: <code>examples/tyumen_dashboard_refined.py</code>. Предыдущая версия страницы сохраняется рядом как <code>dashboard_refined_previous.html</code>.</p>
</footer>
</div>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
__MAP_SCRIPT__
</body>
</html>
"""


def render_page() -> str:
    page = PAGE_TEMPLATE
    replacements = {
        "__GENERATED_AT__": html.escape(generated_at),
        "__METRIC_COUNT__": fmt(metric_count, 0),
        "__WATER_COUNT__": fmt(object_meta["water"], 0),
        "__SCHOOL_COUNT__": fmt(object_meta["schools"], 0),
        "__BLOCK_COUNT__": fmt(object_meta["blocks"], 0),
        "__CARDS__": cards_html,
        "__EXPLANATIONS__": explanations_html,
        "__SUMMARY_TABLE__": summary_table,
        "__NUMERIC_TABLE__": numeric_table,
        "__TOP_POPULATION__": top_population_table,
        "__TOP_ACCESSIBILITY__": top_accessibility_table,
        "__TOP_PROVISION__": top_provision_low_table,
        "__PROVISION_TOTALS__": provision_totals_table,
        "__PROVISION_TABLE__": provision_table,
        "__TOP_DEFICIT__": top_deficit_table,
        "__LINKS_TABLE__": links_table,
        "__SCHOOLS_TABLE__": schools_table,
        "__CATEGORICAL_TABLE__": categorical_table,
        "__CLUSTERS_TABLE__": clusters_table,
        "__COVERAGE_TABLE__": coverage_table,
        "__QUALITY_TABLE__": quality_table,
        "__QUALITY_META_TABLE__": quality_meta_table,
        "__BLOCKS_QUALITY_TABLE__": blocks_quality_table,
        "__BLOCKS_QUALITY_STATS_TABLE__": blocks_quality_stats_table,
        "__SPLIT_META_TABLE__": split_meta_table,
        "__SOURCE_META_TABLE__": source_meta_table,
        "__MATRIX_META_TABLE__": matrix_meta_table,
        "__CONCLUSIONS__": conclusions_html,
        "__LIMITATIONS__": limitations_html,
        "__MAP_SCRIPT__": map_script,
    }
    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)
    return page


html_doc = render_page()
if OUTPUT.exists():
    PREVIOUS_OUTPUT.write_bytes(OUTPUT.read_bytes())
OUTPUT.write_text(html_doc, encoding="utf-8")
print(f"Готово: {OUTPUT}")
print(f"Размер: {OUTPUT.stat().st_size / 1_000_000:.1f} МБ")
print(f"Показателей в переключателе: {metric_count}")
print(f"Кварталов в карте: {len(block_data['features'])}")
print(f"Школ на карте: {len(school_data['features'])}")
print("Откройте файл в браузере.")
