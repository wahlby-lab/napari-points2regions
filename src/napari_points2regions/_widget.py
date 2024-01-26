from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import colorcet as cc
import numpy as np
import pandas as pd
from magicgui import magic_factory
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


def _on_load_points_widget_init(widget):
    def refresh_columns(csv_file):
        if (
            csv_file is not None
            and csv_file.is_file()
            and csv_file.suffix.lower() == ".csv"
        ):
            columns = pd.read_csv(csv_file, nrows=0).columns
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

    def on_csv_file_changed(csv_file):
        refresh_columns(csv_file)

    on_csv_file_changed(widget.csv_file.value)
    widget.csv_file.changed.connect(on_csv_file_changed)


@magic_factory(widget_init=_on_load_points_widget_init, call_button="Load")
def load_points(
    csv_file: Annotated[Path, {"mode": "r", "filter": "*.csv"}],
    x_column: Annotated[str, {"widget_type": "ComboBox"}],
    y_column: Annotated[str, {"widget_type": "ComboBox"}],
    label_column: Annotated[str, {"widget_type": "ComboBox"}],
    dataset_column: Annotated[Optional[str], {"widget_type": "ComboBox"}],
    new_layer_name: str = _DEFAULT_LAYER_NAME,
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
    }
    return (points, layer_args, "points")


def _on_points2regions_widget_init(widget):
    def refresh_features(points_layer):
        if points_layer is not None:
            features = list(points_layer.features.columns)
        else:
            features = []
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

    def on_points_layer_changed(points_layer):
        refresh_features(points_layer)
        if points_layer is not None:
            points_layer.events.features.connect(
                # always use latest value of widget.points_layer!
                lambda event: refresh_features(widget.points_layer.value)
            )

    on_points_layer_changed(widget.points_layer.value)
    widget.points_layer.changed.connect(on_points_layer_changed)


@magic_factory(widget_init=_on_points2regions_widget_init, call_button="Run")
def points2regions(
    points_layer: "napari.layers.Points",
    label_feature: Annotated[str, {"widget_type": "ComboBox"}],
    dataset_feature: Annotated[Optional[str], {"widget_type": "ComboBox"}],
    new_region_feature: str = _DEFAULT_REGION_FEATURE,
    num_clusters: int = _DEFAULT_NUM_CLUSTERS,
    pixel_width: float = _DEFAULT_PIXEL_WIDTH,
    pixel_smoothing: float = _DEFAULT_PIXEL_SMOOTHING,
    min_num_pts_per_pixel: float = _DEFAULT_MIN_NUM_PTS_PER_PIXEL,
    seed: int = _DEFAULT_SEED,
):
    p2r = Points2Regions(
        xy=points_layer.data,
        labels=points_layer.features[label_feature],
        pixel_width=pixel_width,
        pixel_smoothing=pixel_smoothing,
        min_num_pts_per_pixel=min_num_pts_per_pixel,
        datasetids=(
            points_layer.features[dataset_feature]
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
    new_features = points_layer.features
    new_features[new_region_feature] = regions
    points_layer.features = new_features  # setter updates color manager
    points_layer.face_color_cycle = _CMAP
    points_layer.face_color = new_region_feature
    if points_layer.features[new_region_feature].nunique() > len(_CMAP):
        show_warning("Identified more regions than available colors")


def _on_adjust_point_display_widget_init(widget):
    def refresh_features(points_layer):
        if points_layer is not None:
            features = list(points_layer.features.columns)
        else:
            features = []
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

    def on_points_layer_changed(points_layer):
        refresh_features(points_layer)
        if points_layer is not None:
            points_layer.events.features.connect(
                # always use latest value of widget.points_layer!
                lambda event: refresh_features(widget.points_layer.value)
            )

    on_points_layer_changed(widget.points_layer.value)
    widget.points_layer.changed.connect(on_points_layer_changed)


@magic_factory(
    widget_init=_on_adjust_point_display_widget_init, call_button="Adjust"
)
def adjust_point_display(
    points_layer: "napari.layers.Points",
    region_feature: Annotated[str, {"widget_type": "ComboBox"}],
    point_size: int = _DEFAULT_POINT_SIZE,
) -> None:
    points_layer.size = point_size
    points_layer.face_color = region_feature
    if points_layer.features[region_feature].nunique() > len(_CMAP):
        show_warning("Identified more regions than available colors")
