#!/usr/bin/env python3
"""Independent validation of the R Scalability/Usability workflow.

This validator does not import, source, or call the R scoring implementation.
It independently recomputes scores, ranks, balanced resource summaries, and
then inspects the actual PDF/SVG/PNG renderings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable
from xml.etree import ElementTree as ET

import fitz
import numpy as np
import pandas as pd
from PIL import Image


ATOL = 1e-11
EXCLUDED_KEYS = {"regionvelocity", "phylovelo"}
CATEGORIES = [
    "Installability",
    "Code quality",
    "Usage guidance",
    "Maintenance",
    "Publication quality",
]
CATEGORY_WEIGHTS = {
    "Installability": 0.25,
    "Code quality": 0.25,
    "Usage guidance": 0.20,
    "Maintenance": 0.15,
    "Publication quality": 0.15,
}
EXPECTED_CATEGORY_COUNTS = {
    "Installability": 6,
    "Code quality": 5,
    "Usage guidance": 5,
    "Maintenance": 5,
    "Publication quality": 3,
}
KNOWN_STRUCTURES = [
    "linear-simple",
    "bifurcating",
    "consecutive-bifurcating",
    "bifurcating-loop",
    "trifurcating",
    "cycle-simple",
    "disconnected",
]
CONDITIONS = {
    "1k_x_10k": ("top_scRNA", 1000, 10000),
    "10k_x_1k": ("top_scRNA", 10000, 1000),
    "200k_x_1k": ("top_scRNA", 200000, 1000),
    "genes_2000": ("velocity_genes_2000", 1000, 20000),
    "genes_5000": ("velocity_genes_5000", 1000, 20000),
}


def normalize_method(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")


def assert_close(
    expected: Iterable[object],
    observed: Iterable[object],
    label: str,
    *,
    atol: float = ATOL,
) -> None:
    np.testing.assert_allclose(
        np.asarray(list(expected), dtype=float),
        np.asarray(list(observed), dtype=float),
        rtol=ATOL,
        atol=atol,
        equal_nan=True,
        err_msg=label,
    )


def stable_descending_order(
    values: pd.Series, methods: list[str]
) -> list[str]:
    return sorted(
        methods,
        key=lambda method: (
            pd.isna(values.loc[method]),
            -float(values.loc[method])
            if pd.notna(values.loc[method])
            else 0.0,
            methods.index(method),
        ),
    )


def style_value(style: str, key: str) -> str | None:
    match = re.search(
        rf"(?:^|;)\s*{re.escape(key)}:\s*([^;]+)", style or ""
    )
    return match.group(1).strip() if match else None


def rgb_hex(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"#[0-9A-Fa-f]{6}", value)
    return match.group(0).upper() if match else value.upper()


def svg_elements(path: Path, tag: str) -> list[dict[str, str]]:
    root = ET.parse(path).getroot()
    return [
        element.attrib
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == tag
    ]


def recompute_resource_scores(
    resource: pd.DataFrame, methods: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = resource.copy()
    data["method_key"] = data["method"].map(normalize_method)
    method_keys = {normalize_method(method) for method in methods}
    assert not set(data["method_key"]) & EXCLUDED_KEYS
    assert set(data["method_key"]) <= method_keys
    top = data[
        data["method"].isin(methods)
        & data["protocol"].eq("top_scRNA")
        & data["measure"].isin(["runtime_seconds", "ram_gb"])
    ].copy()
    assert top["dataset_id"].nunique() == 68
    assert len(top) == 28 * 68 * 2
    top["directed"] = -pd.to_numeric(top["clean_value"], errors="coerce")
    groups = top.groupby(["measure", "dataset_id"], dropna=False)[
        "directed"
    ]
    top["task_n_valid_python"] = groups.transform("count")
    top["task_mean_python"] = groups.transform("mean")
    top["task_sd_python"] = groups.transform("std")
    valid = (
        top["directed"].notna()
        & top["task_n_valid_python"].ge(2)
        & top["task_sd_python"].notna()
        & top["task_sd_python"].gt(0)
    )
    top["z_python"] = np.nan
    top.loc[valid, "z_python"] = (
        top.loc[valid, "directed"]
        - top.loc[valid, "task_mean_python"]
    ) / top.loc[valid, "task_sd_python"]

    means = (
        top.groupby(["method", "measure"])["z_python"]
        .mean()
        .unstack()
        .reindex(methods)
    )
    counts = (
        top.groupby(["method", "measure"])["z_python"]
        .count()
        .unstack()
        .reindex(methods)
    )
    components = means.notna().sum(axis=1)
    expected = pd.DataFrame(index=methods)
    expected["T"] = means["runtime_seconds"]
    expected["R"] = means["ram_gb"]
    expected["Sc"] = means.mean(axis=1, skipna=True)
    expected.loc[components.eq(0), "Sc"] = np.nan
    expected["T_n_valid"] = counts["runtime_seconds"]
    expected["R_n_valid"] = counts["ram_gb"]
    expected["Sc_n_components"] = components
    expected["effective_weight_T"] = np.where(
        expected["T"].notna(), 1 / components, np.nan
    )
    expected["effective_weight_R"] = np.where(
        expected["R"].notna(), 1 / components, np.nan
    )
    expected["Sc_rank"] = expected["Sc"].rank(
        ascending=False, method="average"
    )
    expected["Sc_reversed_rank"] = 29 - expected["Sc_rank"]
    return top, expected


def recompute_usability(
    items: pd.DataFrame, rules: pd.DataFrame, methods: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    assert len(rules) == 24
    assert not rules.duplicated(["category", "subcategory"]).any()
    assert set(rules["category"]) == set(CATEGORIES)
    for category in CATEGORIES:
        rule = rules[rules["category"].eq(category)]
        assert len(rule) == EXPECTED_CATEGORY_COUNTS[category]
        assert math.isclose(
            float(rule["subcategory_weight"].sum()),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        )
        assert_close(
            rule["category_weight"],
            np.repeat(CATEGORY_WEIGHTS[category], len(rule)),
            f"S7 category weight {category}",
        )

    data = items[items["method"].isin(methods)].copy()
    assert len(data) == 28 * 24
    assert not data.duplicated(
        ["method", "category", "subcategory"]
    ).any()
    assert data["subscore"].dropna().between(0, 10).all()
    data["contribution_python"] = (
        data["subscore"] * data["subcategory_weight"]
    )
    category_rows: list[dict[str, object]] = []
    for method in methods:
        for category in CATEGORIES:
            observed = data[
                data["method"].eq(method)
                & data["category"].eq(category)
            ].copy()
            rule = rules[rules["category"].eq(category)].copy()
            merged = rule[
                ["subcategory", "subcategory_weight"]
            ].merge(
                observed[
                    [
                        "subcategory",
                        "subcategory_weight",
                        "subscore",
                        "contribution_python",
                    ]
                ],
                on="subcategory",
                how="outer",
                suffixes=("_rule", "_item"),
                indicator=True,
            )
            complete = (
                len(observed) == EXPECTED_CATEGORY_COUNTS[category]
                and (merged["_merge"] == "both").all()
                and merged["subscore"].notna().all()
                and np.allclose(
                    merged["subcategory_weight_rule"],
                    merged["subcategory_weight_item"],
                    rtol=0,
                    atol=1e-12,
                )
                and math.isclose(
                    float(observed["subcategory_weight"].sum()),
                    1.0,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            )
            category_rows.append(
                {
                    "method": method,
                    "category": category,
                    "score_python": (
                        float(observed["contribution_python"].sum())
                        if complete
                        else np.nan
                    ),
                    "complete_python": complete,
                }
            )
    categories = pd.DataFrame(category_rows)
    matrix = categories.pivot(
        index="method", columns="category", values="score_python"
    ).reindex(methods)
    complete = matrix.notna().all(axis=1)
    u = sum(
        matrix[category] * weight
        for category, weight in CATEGORY_WEIGHTS.items()
    )
    u.loc[~complete] = np.nan
    return categories, u


def condition_rows(
    resource: pd.DataFrame, measure: str, condition: str
) -> pd.DataFrame:
    protocol, cells, genes = CONDITIONS[condition]
    return resource[
        resource["measure"].eq(measure)
        & resource["protocol"].eq(protocol)
        & resource["cells"].eq(cells)
        & resource["genes"].eq(genes)
    ].copy()


def verify_balanced_costs(
    resource: pd.DataFrame,
    heat: pd.DataFrame,
    result: Path,
) -> None:
    data = resource.copy()
    data["display_python"] = np.where(
        data["measure"].eq("runtime_seconds"),
        data["clean_value"] / 60,
        data["clean_value"],
    )
    for row in heat.itertuples(index=False):
        rows = condition_rows(data, row.measure, row.condition)
        assert len(rows) > 0
        method_rows = rows[rows["method"].eq(row.method)]
        balanced = method_rows.groupby("structure", dropna=False)[
            "display_python"
        ].mean()
        expected = (
            float(balanced.mean()) if balanced.notna().any() else np.nan
        )
        assert_close([expected], [row.value], "balanced heatmap value")
        assert int(row.n_balance_expected) == rows["structure"].nunique()
        assert int(row.n_balance_valid) == int(balanced.notna().sum())
        assert int(row.n_records_expected) == rows["dataset_id"].nunique()
        assert int(row.n_records_numeric) == int(
            method_rows["display_python"].notna().sum()
        )

    source = read_csv(
        result / "tables/scalability_heatmap_source_detail.csv"
    )
    assert len(source) == 28 * 29 * 2
    runtime = source["measure"].eq("runtime_seconds")
    assert_close(
        source.loc[runtime, "display_value"],
        source.loc[runtime, "raw_cost_value"] / 60,
        "seconds converted to minutes exactly once",
    )
    assert_close(
        source.loc[~runtime, "display_value"],
        source.loc[~runtime, "raw_cost_value"],
        "RAM remains GB",
    )
    assert source["source_path"].notna().all()
    assert source["source_locator"].notna().all()


def verify_structure_time(
    resource: pd.DataFrame, observed: pd.DataFrame
) -> None:
    rows = resource[
        resource["protocol"].eq("top_scRNA")
        & resource["measure"].eq("runtime_seconds")
        & resource["structure"].isin(KNOWN_STRUCTURES)
    ].copy()
    rows["minutes"] = rows["clean_value"] / 60
    for row in observed.itertuples(index=False):
        subset = rows[
            rows["method"].eq(row.method)
            & rows["structure"].eq(row.condition)
        ]
        scale_means = subset.groupby(["cells", "genes"])[
            "minutes"
        ].mean()
        expected = (
            float(scale_means.mean())
            if scale_means.notna().any()
            else np.nan
        )
        assert_close([expected], [row.value], "structure-balanced time")


def pdf_method_order(
    page: fitz.Page,
    methods: list[str],
    *,
    x_min: float | None = None,
    x_max: float | None = None,
) -> list[str]:
    hits: list[tuple[float, float, str]] = []
    for word in page.get_text("words"):
        text = word[4]
        center_x = (word[0] + word[2]) / 2
        if text not in methods:
            continue
        if x_min is not None and center_x < x_min:
            continue
        if x_max is not None and center_x >= x_max:
            continue
        hits.append((word[1], word[0], text))
    hits.sort()
    return [text for _, _, text in hits]


def verify_bar_rectangles(
    svg_path: Path,
    geometry: pd.DataFrame,
    panel_height_in: float,
) -> int:
    rects = svg_elements(svg_path, "rect")
    actual: list[dict[str, object]] = []
    for rect in rects:
        style = rect.get("style", "")
        if rgb_hex(style_value(style, "stroke")) != "#4D4D4D":
            continue
        try:
            actual.append(
                {
                    "x": float(rect["x"]),
                    "y": float(rect["y"]),
                    "width": float(rect["width"]),
                    "height": float(rect["height"]),
                    "fill": rgb_hex(style_value(style, "fill")),
                }
            )
        except (KeyError, ValueError):
            continue
    assert len(actual) >= len(geometry)
    for row in geometry.itertuples(index=False):
        x = float(row.x_left_in) * 72
        width = float(row.plotted_width_in) * 72
        height = float(row.bar_height_in) * 72
        y = (
            panel_height_in
            - float(row.y_center_in)
            - float(row.bar_height_in) / 2
        ) * 72
        candidates = [
            rect
            for rect in actual
            if abs(float(rect["x"]) - x) < 0.15
            and abs(float(rect["y"]) - y) < 0.15
        ]
        assert len(candidates) == 1, (row.method, candidates)
        rect = candidates[0]
        assert abs(float(rect["width"]) - width) < 0.15
        assert abs(float(rect["height"]) - height) < 0.15
        assert rect["fill"] == rgb_hex(str(row.fill_hex))
    return len(geometry)


def verify_heat_rectangles(
    svg_path: Path,
    geometry: pd.DataFrame,
    panel_height_in: float,
) -> int:
    rects = svg_elements(svg_path, "rect")
    candidates: list[dict[str, object]] = []
    for rect in rects:
        style = rect.get("style", "")
        if rgb_hex(style_value(style, "stroke")) != "#FFFFFF":
            continue
        try:
            candidates.append(
                {
                    "x": float(rect["x"]),
                    "y": float(rect["y"]),
                    "width": float(rect["width"]),
                    "height": float(rect["height"]),
                    "fill": rgb_hex(style_value(style, "fill")),
                }
            )
        except (KeyError, ValueError):
            continue
    for row in geometry.itertuples(index=False):
        x = (float(row.x_center_in) - float(row.width_in) / 2) * 72
        y = (
            panel_height_in
            - float(row.y_center_in)
            - float(row.height_in) / 2
        ) * 72
        found = [
            rect
            for rect in candidates
            if abs(float(rect["x"]) - x) < 0.15
            and abs(float(rect["y"]) - y) < 0.15
        ]
        assert len(found) == 1, (row.method, row.measure, row.condition)
        rect = found[0]
        assert abs(float(rect["width"]) - float(row.width_in) * 72) < 0.15
        assert abs(float(rect["height"]) - float(row.height_in) * 72) < 0.15
        assert rect["fill"] == rgb_hex(str(row.fill_hex))
    return len(geometry)


def verify_cost_labels(
    pdf_path: Path, geometry: pd.DataFrame, panel_height_in: float
) -> int:
    with fitz.open(pdf_path) as document:
        words = document[0].get_text("words")
    labels = [
        word
        for word in words
        if re.fullmatch(r"(?:-?[0-9]+\.[0-9]{2}|NA)", word[4])
    ]
    for row in geometry.itertuples(index=False):
        expected_label = (
            "NA"
            if pd.isna(row.label)
            else (
                str(row.label)
                if isinstance(row.label, str)
                else f"{float(row.label):.2f}"
            )
        )
        expected_x = float(row.x_center_in) * 72
        expected_y = (panel_height_in - float(row.y_center_in)) * 72
        chosen = min(
            labels,
            key=lambda word: abs((word[0] + word[2]) / 2 - expected_x)
            + abs((word[1] + word[3]) / 2 - expected_y),
        )
        assert abs((chosen[0] + chosen[2]) / 2 - expected_x) < 2.0
        assert abs((chosen[1] + chosen[3]) / 2 - expected_y) < 3.0
        assert chosen[4] == expected_label
    return len(geometry)


def verify_bubbles(
    svg_path: Path,
    layer: pd.DataFrame,
    u_order: list[str],
) -> dict[str, object]:
    circles = svg_elements(svg_path, "circle")
    y_values = sorted(
        {round(float(circle["cy"]), 2) for circle in circles}
    )
    counts = {
        y: sum(
            round(float(circle["cy"]), 2) == y for circle in circles
        )
        for y in y_values
    }
    row_ys = [y for y in y_values if counts[y] == 5]
    assert len(row_ys) == 28
    body = [
        circle
        for circle in circles
        if round(float(circle["cy"]), 2) in row_ys
    ]
    assert len(body) == 140
    columns = sorted(
        {round(float(circle["cx"]), 2) for circle in body}
    )
    assert len(columns) == 5

    for column in columns:
        points = sorted(
            [
                circle
                for circle in body
                if round(float(circle["cx"]), 2) == column
            ],
            key=lambda circle: float(circle["cy"]),
        )
        for first, second in zip(points, points[1:]):
            gap = float(second["cy"]) - float(first["cy"])
            assert (
                gap + 0.05
                >= float(first["r"]) + float(second["r"])
            ), "Bubble overlap"

    radii = [float(circle["r"]) for circle in body]
    intercept = min(radii) - (max(radii) - min(radii)) / 14 * 2
    slope = (max(radii) - min(radii)) / 14
    for i, method in enumerate(u_order):
        for j, category in enumerate(CATEGORIES):
            circle = next(
                item
                for item in body
                if round(float(item["cy"]), 2) == row_ys[i]
                and round(float(item["cx"]), 2) == columns[j]
            )
            expected = layer[
                layer["method"].eq(method)
                & layer["category"].eq(category)
            ].iloc[0]
            fill = rgb_hex(style_value(circle.get("style", ""), "fill"))
            assert fill == str(expected["fill_hex"]).upper()
            assert (
                abs(
                    float(circle["r"])
                    - (slope * float(expected["size_mm"]) + intercept)
                )
                < 0.02
            )
    return {
        "body_bubbles": 140,
        "row_count": 28,
        "column_count": 5,
        "no_vertical_overlap": True,
    }


def verify_figures(
    result: Path,
    methods: list[str],
    sc_order: list[str],
    u_order: list[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    registry = read_csv(result / "figures/figure_registry.csv")
    assert len(registry) == 7
    file_checks: list[dict[str, object]] = []
    for spec in registry.itertuples(index=False):
        stem = result / "figures" / str(spec.figure)
        pdf = stem.with_suffix(".pdf")
        svg = stem.with_suffix(".svg")
        png = stem.with_suffix(".png")
        plotting_paths = str(spec.plotting_csv).split(";")
        assert all((result / path).is_file() for path in plotting_paths)
        for path in (pdf, svg, png):
            assert path.is_file() and path.stat().st_size > 0

        with fitz.open(pdf) as document:
            assert len(document) == 1
            page = document[0]
            assert not page.get_images(full=True)
            fonts = page.get_fonts(full=True)
            assert fonts
            for font in fonts:
                extracted = document.extract_font(font[0])
                assert extracted[3], (spec.figure, font)
            assert abs(page.rect.width - float(spec.width_inches) * 72) < 1
            assert abs(page.rect.height - float(spec.height_inches) * 72) < 1
            spans = [
                span
                for block in page.get_text("dict")["blocks"]
                if "lines" in block
                for line in block["lines"]
                for span in line["spans"]
            ]
            assert all(
                span["bbox"][0] >= -1
                and span["bbox"][1] >= -1
                and span["bbox"][2] <= page.rect.width + 1
                and span["bbox"][3] <= page.rect.height + 1
                for span in spans
            ), spec.figure
            if spec.dimension == "Sc":
                assert pdf_method_order(page, methods) == sc_order
            elif spec.dimension == "U":
                assert pdf_method_order(page, methods) == u_order
            elif spec.dimension == "independent_Sc_and_U":
                split = 10.5 * 72
                right_start = (10.5 + 0.35) * 72
                assert (
                    pdf_method_order(page, methods, x_max=split)
                    == sc_order
                )
                assert (
                    pdf_method_order(page, methods, x_min=right_start)
                    == u_order
                )

        root = ET.parse(svg).getroot()
        assert not any(
            item.tag.rsplit("}", 1)[-1] == "image"
            for item in root.iter()
        )
        assert any(
            item.tag.rsplit("}", 1)[-1] == "text"
            for item in root.iter()
        )
        width = float(re.sub(r"pt$", "", root.attrib["width"]))
        height = float(re.sub(r"pt$", "", root.attrib["height"]))
        assert abs(width - float(spec.width_inches) * 72) < 1
        assert abs(height - float(spec.height_inches) * 72) < 1

        with Image.open(png) as image:
            image.load()
            dpi = image.info.get("dpi", (0.0, 0.0))
            assert min(float(dpi[0]), float(dpi[1])) >= 299
            assert abs(image.width - float(spec.width_inches) * 300) <= 1
            assert abs(image.height - float(spec.height_inches) * 300) <= 1

        file_checks.append(
            {
                "figure": spec.figure,
                "dimension": spec.dimension,
                "pdf_raster_images": 0,
                "pdf_fonts_embedded": True,
                "svg_has_editable_text": True,
                "svg_has_image": False,
                "png_full_decode": True,
                "png_dpi": [float(dpi[0]), float(dpi[1])],
                "method_order_verified": True,
                "text_inside_canvas": True,
            }
        )

    bar_geometry = read_csv(
        result / "tables/main_bar_render_geometry.csv"
    )
    bars_checked = verify_bar_rectangles(
        result / "figures/FigureA_Scalability_independent_rank.svg",
        bar_geometry[bar_geometry["dimension"].eq("Sc")],
        16,
    )
    bars_checked += verify_bar_rectangles(
        result / "figures/FigureB_Usability_weighted_rank.svg",
        bar_geometry[bar_geometry["dimension"].eq("U")],
        16,
    )
    heat_geometry = read_csv(
        result / "tables/scalability_heatmap_render_geometry.csv"
    )
    heat_rectangles = verify_heat_rectangles(
        result / "figures/FigureA_Scalability_independent_rank.svg",
        heat_geometry,
        16,
    )
    heat_labels = verify_cost_labels(
        result / "figures/FigureA_Scalability_independent_rank.pdf",
        heat_geometry,
        16,
    )
    layer = read_csv(
        result / "tables/usability_actual_ggplot_layer.csv"
    )
    assert len(layer) == 140
    assert layer["shape"].eq(21).all()
    assert_close([layer["size_mm"].min()], [2], "minimum bubble size")
    assert_close([layer["size_mm"].max()], [16], "maximum bubble size")
    bubbles = verify_bubbles(
        result / "figures/FigureB_Usability_weighted_rank.svg",
        layer,
        u_order,
    )
    rendered = {
        "bars_checked": bars_checked,
        "heatmap_rectangles_checked": heat_rectangles,
        "heatmap_labels_checked": heat_labels,
        **bubbles,
    }
    return file_checks, rendered


def verify(project: Path, result: Path) -> dict[str, object]:
    manifest = read_csv(project / "config/input_sha256.csv")
    manifest_checks = 0
    for row in manifest.itertuples(index=False):
        path = project / str(row.relative_path)
        if path.is_file():
            assert path.stat().st_size == int(row.bytes)
            assert sha256(path) == row.sha256
            manifest_checks += 1
        elif str(row.scoring_or_rule_input).lower() == "true":
            raise AssertionError(f"Missing locked input: {row.relative_path}")

    palette = read_csv(project / "data/method_colors.csv").sort_values(
        "method_order"
    )
    methods = palette["method"].tolist()
    assert len(methods) == 28
    assert palette["method_order"].tolist() == list(range(1, 29))
    assert len({normalize_method(method) for method in methods}) == 28
    assert not {
        normalize_method(method) for method in methods
    } & EXCLUDED_KEYS

    resource = read_csv(project / "data/resource_clean_long.csv")
    top, expected_sc = recompute_resource_scores(resource, methods)
    observed_z = read_csv(result / "tables/scalability_task_z.csv")
    keys = ["method", "measure", "dataset_id"]
    merged_z = top.merge(
        observed_z[
            keys
            + [
                "z",
                "task_n_valid",
                "task_mean_directed",
                "task_sd",
            ]
        ],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    assert (merged_z["_merge"] == "both").all()
    assert_close(merged_z["z_python"], merged_z["z"], "R task z")
    assert_close(
        merged_z["task_n_valid_python"],
        merged_z["task_n_valid"],
        "task valid n",
    )
    assert_close(
        merged_z["task_mean_python"],
        merged_z["task_mean_directed"],
        "task mean",
    )
    assert_close(
        merged_z["task_sd_python"],
        merged_z["task_sd"],
        "task sample SD",
    )

    observed_sc = (
        read_csv(result / "tables/scalability_independent_rank.csv")
        .set_index("method")
        .reindex(methods)
    )
    for column in [
        "T",
        "R",
        "Sc",
        "T_n_valid",
        "R_n_valid",
        "Sc_n_components",
        "effective_weight_T",
        "effective_weight_R",
        "Sc_rank",
        "Sc_reversed_rank",
    ]:
        assert_close(
            expected_sc[column],
            observed_sc[column],
            f"Sc {column}",
        )
    sc_order = stable_descending_order(expected_sc["Sc"], methods)
    observed_sc_order = read_csv(result / "tables/scalability_order.csv")
    assert observed_sc_order["method"].tolist() == sc_order
    assert observed_sc_order["display_order"].tolist() == list(
        range(1, 29)
    )

    items = read_csv(
        project / "data/usability_item_contributions.csv"
    )
    rules = read_csv(project / "data/usability_rules_S7.csv")
    expected_categories, expected_u = recompute_usability(
        items, rules, methods
    )
    observed_categories = read_csv(
        result / "tables/usability_categories_recomputed.csv"
    )
    category_merge = expected_categories.merge(
        observed_categories[
            ["method", "category", "score", "category_complete"]
        ],
        on=["method", "category"],
        validate="one_to_one",
    )
    assert_close(
        category_merge["score_python"],
        category_merge["score"],
        "U category scores",
    )
    assert (
        category_merge["complete_python"].astype(bool).to_numpy()
        == category_merge["category_complete"].astype(bool).to_numpy()
    ).all()
    observed_items = read_csv(
        result / "tables/usability_item_contributions_recomputed.csv"
    )
    item_merge = items.merge(
        observed_items[
            [
                "method",
                "category",
                "subcategory",
                "recomputed_contribution",
            ]
        ],
        on=["method", "category", "subcategory"],
        validate="one_to_one",
    )
    assert_close(
        item_merge["subscore"] * item_merge["subcategory_weight"],
        item_merge["recomputed_contribution"],
        "item contributions",
    )
    expected_u_rank = expected_u.rank(
        ascending=False, method="average"
    )
    observed_u = (
        read_csv(result / "tables/usability_independent_rank.csv")
        .set_index("method")
        .reindex(methods)
    )
    assert_close(expected_u, observed_u["U"], "weighted U")
    assert_close(expected_u_rank, observed_u["U_rank"], "U rank")
    u_order = stable_descending_order(expected_u, methods)
    observed_u_order = read_csv(result / "tables/usability_order.csv")
    assert observed_u_order["method"].tolist() == u_order
    assert observed_u_order["display_order"].tolist() == list(
        range(1, 29)
    )
    assert sc_order != u_order

    # This public workflow validates against the standalone inputs directly.
    # Previous-package scores are not required and are not calculation inputs.

    heat = read_csv(result / "tables/scalability_heatmaps.csv")
    assert len(heat) == 28 * 10
    verify_balanced_costs(resource, heat, result)
    scope = read_csv(result / "tables/resource_scope_audit.csv")
    extra_scope = scope[scope["condition"].eq("200k_x_1k")]
    assert len(extra_scope) == 2
    assert extra_scope["n_unique_tasks"].eq(1).all()
    assert extra_scope["n_structures"].eq(1).all()
    assert extra_scope["structures"].eq("extra-large").all()
    velocity = scope[
        scope["condition"].isin(["genes_2000", "genes_5000"])
    ]
    assert velocity["cells"].astype(str).eq("1000").all()
    assert velocity["genes"].astype(str).eq("20000").all()

    structure_time = read_csv(
        result / "tables/structure_mean_time.csv"
    )
    assert len(structure_time) == 28 * 7
    verify_structure_time(resource, structure_time)
    structure_scope = read_csv(
        result / "tables/structure_scope_audit.csv"
    )
    assert set(
        structure_scope.loc[
            structure_scope["included_in_structure_figure"], "structure"
        ]
    ) == set(KNOWN_STRUCTURES)
    assert (
        ~structure_scope.loc[
            structure_scope["scope"].eq("extra_large"),
            "included_in_structure_figure",
        ]
    ).all()

    vram = read_csv(result / "tables/vram_source_audit.csv")
    assert len(vram) == len(
        resource[resource["measure"].eq("vram_gb")]
    )
    registry_text = (
        result / "figures/figure_registry.csv"
    ).read_text(encoding="utf-8").lower()
    assert "vram" not in registry_text

    zero_rows = resource[
        resource["clean_value"].eq(0)
        & resource["measure"].isin(["runtime_seconds", "ram_gb"])
    ]
    if len(zero_rows):
        zero_keys = zero_rows[
            zero_rows["protocol"].eq("top_scRNA")
        ][keys]
        matched_zero = zero_keys.merge(observed_z[keys], on=keys)
        assert len(matched_zero) == len(zero_keys)

    file_checks, rendered = verify_figures(
        result, methods, sc_order, u_order
    )

    sc_ties = (
        expected_sc.reset_index(names="method")
        .groupby("Sc_rank")["method"]
        .agg(list)
    )
    sc_ties = [group for group in sc_ties if len(group) > 1]
    u_ties = (
        pd.DataFrame(
            {"method": methods, "rank": expected_u_rank.values}
        )
        .groupby("rank")["method"]
        .agg(list)
    )
    u_ties = [group for group in u_ties if len(group) > 1]

    summary: dict[str, object] = {
        "status": "PASS",
        "validator": "independent_python_recalculation_and_render_audit_v2",
        "input_hashes_verified": manifest_checks,
        "fixed_methods": 28,
        "excluded_method_keys": sorted(EXCLUDED_KEYS),
        "resource_rows": int(len(resource)),
        "top_scRNA_cost_rows": int(len(top)),
        "top_scRNA_tasks_per_measure": 68,
        "valid_task_z_rows": int(top["z_python"].notna().sum()),
        "usability_items": int(len(items)),
        "category_cells": int(len(expected_categories)),
        "heatmap_summary_cells": int(len(heat)),
        "heatmap_source_rows": int(
            len(
                read_csv(
                    result
                    / "tables/scalability_heatmap_source_detail.csv"
                )
            )
        ),
        "sc_top5": sc_order[:5],
        "u_top5": u_order[:5],
        "sc_exact_ties": sc_ties,
        "u_exact_ties": u_ties,
        "sc_and_u_orders_differ": True,
        "figures_checked": len(file_checks),
        "rendered_glyph_checks": rendered,
        "file_checks": file_checks,
    }
    validation = result / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    (
        validation / "independent_python_checks.json"
    ).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(file_checks).to_csv(
        validation / "figure_file_checks.csv", index=False
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key != "file_checks"
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="R output root; defaults to PROJECT/results",
    )
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    result = (
        args.result.expanduser().resolve()
        if args.result is not None
        else project / "results"
    )
    verify(project, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
