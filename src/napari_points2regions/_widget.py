from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import colorcet as cc
import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.layers.utils.layer_utils import features_to_pandas_dataframe
from napari.utils.notifications import show_warning
from points2regions import Points2Regions

if TYPE_CHECKING:
    import napari


_CMAP = cc.glasbey

_DEFAULT_X_COLUMN = "x"
_DEFAULT_Y_COLUMN = "y"
_DEFAULT_LABEL_COLUMN = "label"
_DEFAULT_DATASET_COLUMN = "dataset"

_DEFAULT_LAYER_NAME = "Points"
_DEFAULT_LABEL_FEATURE = "label"
_DEFAULT_DATASET_FEATURE = "dataset"
_DEFAULT_REGION_FEATURE = "region"

_DEFAULT_NUM_CLUSTERS = 8
_DEFAULT_PIXEL_WIDTH = 1
_DEFAULT_PIXEL_SMOOTHING = 5
_DEFAULT_MIN_NUM_PTS_PER_PIXEL = 0
_DEFAULT_SEED = 42

_DEFAULT_POINT_SIZE = 10

_DEFAULT_CSV_FILE = Path().home() / "features.csv"


def _on_load_points_widget_init(widget):
    def on_csv_file_changed(csv_file):
        if (
            csv_file is not None
            and csv_file.is_file()
            and csv_file.suffix.lower() == ".csv"
        ):
            columns = pd.read_csv(csv_file, nrows=0).columns
            if widget.new_layer_name.value == _DEFAULT_LAYER_NAME:
                widget.new_layer_name.value = csv_file.stem
        else:
            columns = []
        x_column = None
        y_column = None
        label_column = None
        dataset_column = None
        remaining_columns = list(columns)
        for column in columns:
            if column.lower() == _DEFAULT_X_COLUMN:
                x_column = column
                remaining_columns.remove(column)
            elif column.lower() == _DEFAULT_Y_COLUMN:
                y_column = column
                remaining_columns.remove(column)
            elif column.lower() == _DEFAULT_LABEL_COLUMN:
                label_column = column
                remaining_columns.remove(column)
            elif column.lower() == _DEFAULT_DATASET_COLUMN:
                dataset_column = column
                remaining_columns.remove(column)
        if x_column is None and remaining_columns:
            x_column = remaining_columns.pop(0)
        if y_column is None and remaining_columns:
            y_column = remaining_columns.pop(0)
        if label_column is None and remaining_columns:
            label_column = remaining_columns.pop(0)
        # https://github.com/pyapp-kit/magicgui/issues/306
        widget.x_column.choices = lambda cb: columns
        widget.y_column.choices = lambda cb: columns
        widget.label_column.choices = lambda cb: columns
        widget.dataset_column.choices = lambda cb: columns
        if x_column is not None:
            widget.x_column.value = x_column
        if y_column is not None:
            widget.y_column.value = y_column
        if label_column is not None:
            widget.label_column.value = label_column
        widget.dataset_column.value = dataset_column

    on_csv_file_changed(widget.csv_file.value)
    widget.csv_file.changed.connect(on_csv_file_changed)


@magic_factory(widget_init=_on_load_points_widget_init, call_button="Load")
def load_points(
    csv_file: Annotated[Path, {"mode": "r", "filter": "*.csv"}],
    x_column: Annotated[
        str, {"widget_type": "ComboBox"}
    ],  # see _on_load_points_widget_init
    y_column: Annotated[
        str, {"widget_type": "ComboBox"}
    ],  # see _on_load_points_widget_init
    label_column: Annotated[
        str, {"widget_type": "ComboBox"}
    ],  # see _on_load_points_widget_init
    dataset_column: Annotated[
        Optional[str], {"widget_type": "ComboBox"}
    ],  # see _on_load_points_widget_init
    new_layer_name: str = _DEFAULT_LAYER_NAME,  # see _on_load_points_widget_init
    new_label_feature: str = _DEFAULT_LABEL_FEATURE,
    new_dataset_feature: str = _DEFAULT_DATASET_FEATURE,
) -> "napari.types.LayerDataTuple":
    df = pd.read_csv(csv_file)
    feature_columns = [label_column]
    if dataset_column is not None:
        feature_columns.append(dataset_column)
    points = df.loc[:, [y_column, x_column]].to_numpy()
    features = df.loc[:, feature_columns].rename(
        columns={
            label_column: new_label_feature,
            dataset_column: new_dataset_feature,
        }
    )
    layer_args = {
        "name": new_layer_name,
        "features": features,
        "size": _DEFAULT_POINT_SIZE,
        "edge_width": 0,
        "metadata": {"csv_file": csv_file},
    }
    return (points, layer_args, "points")


def _on_points2regions_widget_init(widget):
    def refresh_features(layer):
        features = list(layer.features.columns) if layer is not None else []
        label_feature = None
        dataset_feature = None
        remaining_features = list(features)
        for feature in features:
            if feature.lower() == _DEFAULT_LABEL_FEATURE:
                label_feature = feature
                remaining_features.remove(feature)
            elif feature.lower() == _DEFAULT_DATASET_FEATURE:
                dataset_feature = feature
                remaining_features.remove(feature)
        if label_feature is None and remaining_features:
            label_feature = remaining_features.pop(0)
        # https://github.com/pyapp-kit/magicgui/issues/306
        widget.label_feature.choices = lambda cb: features
        widget.dataset_feature.choices = lambda cb: features
        if label_feature is not None:
            widget.label_feature.value = label_feature
        widget.dataset_feature.value = dataset_feature

    def on_layer_changed(layer):
        refresh_features(layer)
        if layer is not None:
            layer.events.features.connect(
                # always use latest value of widget.layer!
                lambda event: refresh_features(widget.layer.value)
            )

    on_layer_changed(widget.layer.value)
    widget.layer.changed.connect(on_layer_changed)


@magic_factory(widget_init=_on_points2regions_widget_init, call_button="Run")
def points2regions(
    layer: "napari.layers.Points",
    label_feature: Annotated[
        str, {"widget_type": "ComboBox"}
    ],  # see _on_points2regions_widget_init
    dataset_feature: Annotated[
        Optional[str], {"widget_type": "ComboBox"}
    ],  # see _on_points2regions_widget_init
    num_clusters: int = _DEFAULT_NUM_CLUSTERS,
    pixel_width: float = _DEFAULT_PIXEL_WIDTH,
    pixel_smoothing: float = _DEFAULT_PIXEL_SMOOTHING,
    min_num_pts_per_pixel: float = _DEFAULT_MIN_NUM_PTS_PER_PIXEL,
    seed: int = _DEFAULT_SEED,
    new_region_feature: str = _DEFAULT_REGION_FEATURE,
):
    p2r = Points2Regions(
        xy=layer.data,
        labels=layer.features[label_feature],
        pixel_width=pixel_width,
        pixel_smoothing=pixel_smoothing,
        min_num_pts_per_pixel=min_num_pts_per_pixel,
        datasetids=(
            layer.features[dataset_feature]
            if dataset_feature is not None
            else None
        ),
    )
    regions = p2r.fit_predict(
        num_clusters=num_clusters,
        seed=seed,
        output="marker",
    )
    regions = (regions + 1).astype(np.uint32)
    new_features = layer.features
    new_features[new_region_feature] = regions
    layer.features = new_features  # setter updates color manager
    layer.face_color_cycle = _CMAP
    layer.face_color = new_region_feature
    if layer.features[new_region_feature].nunique() > len(_CMAP):
        show_warning("Identified more regions than available colors")


def _on_adjust_point_display_widget_init(widget):
    def refresh_features(layer):
        features = list(layer.features.columns) if layer is not None else []
        region_feature = None
        remaining_features = list(features)
        for feature in features:
            if feature.lower() == _DEFAULT_REGION_FEATURE:
                region_feature = feature
                remaining_features.remove(feature)
        if region_feature is None and remaining_features:
            region_feature = remaining_features.pop(-1)
        # https://github.com/pyapp-kit/magicgui/issues/306
        widget.region_feature.choices = lambda cb: features
        if region_feature is not None:
            widget.region_feature.value = region_feature

    def on_layer_changed(layer):
        refresh_features(layer)
        if layer is not None:
            layer.events.features.connect(
                # always use latest value of widget.layer!
                lambda event: refresh_features(widget.layer.value)
            )

    on_layer_changed(widget.layer.value)
    widget.layer.changed.connect(on_layer_changed)


@magic_factory(
    widget_init=_on_adjust_point_display_widget_init, call_button="Adjust"
)
def adjust_point_display(
    layer: "napari.layers.Points",
    region_feature: Annotated[
        str, {"widget_type": "ComboBox"}
    ],  # see _on_adjust_point_display_widget_init
    point_size: int = _DEFAULT_POINT_SIZE,
):
    layer.size = point_size
    layer.face_color = region_feature
    if layer.features[region_feature].nunique() > len(_CMAP):
        show_warning("Identified more regions than available colors")


def _on_export_point_features_widget_init(widget):
    def on_layer_changed(layer):
        if (
            layer is not None
            and widget.new_csv_file.value == _DEFAULT_CSV_FILE
        ):
            orig_csv_file = layer.metadata.get("csv_file")
            if orig_csv_file is not None:
                widget.new_csv_file.value = orig_csv_file.with_name(
                    f"{orig_csv_file.stem}_features.csv"
                )
            else:
                widget.new_csv_file.value = (
                    widget.new_csv_file.value.with_name(
                        f"{layer.name}_features.csv"
                    )
                )

    on_layer_changed(widget.layer.value)
    widget.layer.changed.connect(on_layer_changed)


@magic_factory(
    widget_init=_on_export_point_features_widget_init, call_button="Export"
)
def export_point_features(
    layer: "napari.layers.Points",
    new_csv_file: Annotated[
        Path, {"mode": "w", "filter": "*.csv"}
    ] = _DEFAULT_CSV_FILE,  # see _on_export_point_features_widget_init
):
    df = features_to_pandas_dataframe(layer.features).copy()
    df.insert(0, "x", layer.data[:, 1])
    df.insert(1, "y", layer.data[:, 0])
    df.to_csv(new_csv_file, index=False)
