from collections import defaultdict
from itertools import zip_longest
from typing import Union

import numpy as np
from absl import logging
from matplotlib import pyplot as plt

from dst.utils import default_to_regular, nested_defaultdict

DEFAULT_COLOUR = "k"
DEFAULT_LINESTYLE = "-"
DEFAULT_MARKER = "o"

BASE_COLORS = {"b", "g", "r", "c", "m", "y", "k", "w"}
BASE_LINESTYLES = {".", "-.", "--", ":"}


try:
    import seaborn as sns
except ImportError:
    logging.info("Please install seaborn for vizualisation purposes.")


def format_plot(ax, plot_format):
    """Helper function that calls the matplotlib API with specified format data."""

    # x,y axis ticks and ticklabels
    if plot_format.get("ticks", {}):
        x_ticks = plot_format["ticks"].get("x", [])
        y_ticks = plot_format["ticks"].get("y", [])
        if x_ticks:
            ax.set_xticks(x_ticks["vals"])
            if "labels" in plot_format["ticks"]["x"]:
                ax.set_xticklabels(plot_format["ticks"]["x"]["labels"])
        if y_ticks:
            ax.set_yticks(y_ticks["vals"])
            if "labels" in plot_format["ticks"]["y"]:
                ax.set_yticklabels(plot_format["ticks"]["y"]["labels"])
    # tick params for x,y axis
    tick_params = plot_format.get("tick_params", {})
    if tick_params:
        for axs_label in tick_params:
            ax.tick_params(axis=axs_label, **tick_params[axs_label])
    # x,y axis labels
    ax.set_ylabel(**plot_format.get("ylabel", {"ylabel": ""}))
    ax.set_xlabel(**plot_format.get("xlabel", {"xlabel": ""}))
    # plot title
    if plot_format.get("title", {}):
        ax.set_title(plot_format["title"].pop("label"), **plot_format["title"])  #
    # legend
    ax.legend(**plot_format.get("legend", {}))
    # x,y limits
    ax.set_ylim(**plot_format.get("ylim", {}))
    ax.set_xlim(**plot_format.get("xlim", {}))
    if plot_format.get("grid", False):
        ax.grid()
    return ax


def format_plot_nb(ax, plot_format):
    """Helper function that calls the matplotlib API with specified format data."""

    # x,y axis ticks and ticklabels
    if plot_format.get("ticks", {}):
        x_ticks = plot_format["ticks"].get("x", [])
        y_ticks = plot_format["ticks"].get("y", [])
        if x_ticks:
            ax.set_xticks(x_ticks["vals"])
            if "labels" in plot_format["ticks"]["x"]:
                ax.set_xticklabels(plot_format["ticks"]["x"]["labels"])
        if y_ticks:
            ax.set_yticks(y_ticks["vals"])
            if "labels" in plot_format["ticks"]["y"]:
                ax.set_yticklabels(plot_format["ticks"]["y"]["labels"])
    # tick params for x,y axis
    tick_params = plot_format.get("tick_params", {})
    if tick_params:
        for axs_label in tick_params:
            ax.tick_params(axis=axs_label, **tick_params[axs_label])
    # x,y axis labels
    ax.set_ylabel(**plot_format.get("ylabel", {"ylabel": ""}))
    ax.set_xlabel(**plot_format.get("xlabel", {"xlabel": ""}))
    # plot title
    if plot_format.get("title", {}):
        ax.set_title(plot_format["title"].pop("label"), **plot_format["title"])  #
    # legend
    ax.legend(**plot_format.get("legend", {}))
    # x,y limits
    ax.set_ylim(**plot_format.get("ylim", {}))
    ax.set_xlim(**plot_format.get("xlim", {}))
    if plot_format.get("grid", False):
        ax.grid()


def plot_stacked_bar(data_dict: dict, plot_format: dict):
    """Plot stacked bar chart.

    Parameters
    ----------
    data_dict
        Formatted as a depth-2 nested dictionary::

            {
            'outer_key': {'inner_key': int, ....}
            ...
            }

        where the first level keys contain data to be stacked horizontally
        and the inner keys are the xlabels.
    plot_format:
        A nested dictionary to specify how various plot elemenets (e.g., title,
        ticks, legend) should be formatted.
    """

    def format_data(data_dict, keys):
        xlabels = []
        for key in keys:
            xlabels += list(data_dict[key])
        xlabels = sorted(list(set(xlabels)))

        data_store = defaultdict(list)
        for label in xlabels:
            for key in keys:
                if label in data_dict[key]:
                    data_store[key].append(data_dict[key][label])
                else:
                    data_store[key].append(0)
        data_store.update([("xlabels", xlabels)])
        return dict(data_store)

    sns.set_palette("bright")
    fmt_data = format_data(data_dict, list(data_dict.keys()))
    xlabels = fmt_data.pop("xlabels")
    width = plot_format.get("width", 0.35)
    fig, ax = plt.subplots(**plot_format.get("subplot", {}))
    start_pos = None
    v_labels = plot_format.get("stack_order", list(fmt_data.keys()))
    colors = plot_format.get("colors", [None] * len(v_labels))
    orientation = plot_format.get("orientation", "vertical")
    y_coord_scale = plot_format.get("scale", {}).get("y", None)
    x_coord_scale = plot_format.get("scale", {}).get("x", None)
    if x_coord_scale:
        logging.warning("This feature is not implemented!")
    if y_coord_scale:
        y_tick_labels = xlabels
        xlabels = list(range(len(xlabels)))
        xlabels = [i * y_coord_scale for i in xlabels]
    for color, label in zip(colors, v_labels):
        series = fmt_data[label]
        if len(series) == 1:
            series = series[0]
        if orientation == "vertical":
            ax.bar(xlabels, series, width, label=label, bottom=start_pos, color=color)
        else:
            ax.barh(xlabels, series, width, label=label, left=start_pos, color=color)
            if y_coord_scale:
                ax.set_yticks(xlabels)
                ax.set_yticklabels(y_tick_labels)
        if not start_pos:
            start_pos = series
        else:
            if isinstance(series, int) or isinstance(series, float):
                start_pos += series
            else:
                start_pos = [start_pos[i] + series[i] for i in range(len(series))]

    format_plot(ax, plot_format)
    return ax, fig


def plot_grouped_stacked_bar(data_dict: dict, plot_format: dict):
    def format_data(data_dict, keys):

        xlabels, grp_labels = [], []
        for stack_key in keys:
            this_stack_grp_labels = list(data_dict[stack_key].keys())
            for grp_label in this_stack_grp_labels:
                xlabels += list(data_dict[stack_key][grp_label].keys())
            grp_labels += this_stack_grp_labels
        xlabels = sorted(list(set(xlabels)))
        grp_labels = sorted(list(set(grp_labels)))

        data_store = nested_defaultdict(list, depth=2)
        for label in xlabels:
            for stack_key in keys:
                for grp_key in grp_labels:
                    if grp_key in data_dict[stack_key]:
                        if label in data_dict[stack_key][grp_key]:
                            data_store[stack_key][grp_key].append(
                                data_dict[stack_key][grp_key][label]
                            )
                        else:
                            data_store[stack_key][grp_key].append(0)
                    else:
                        data_store[stack_key][grp_key].append(0)
        data_store.update([("xlabels", xlabels)])
        data_store.update([("grp_labels", grp_labels)])

        return default_to_regular(data_store)

    fmt_data = format_data(data_dict, list(data_dict.keys()))
    xlabels = fmt_data.pop("xlabels")
    grp_labels = fmt_data.pop("grp_labels")
    width = plot_format.get("width", 0.35)
    fig, ax = plt.subplots(**plot_format.get("subplot", {}))
    start_pos = {key: {grp_label: None for grp_label in grp_labels} for key in fmt_data}
    v_labels = plot_format.get("stack_order", list(fmt_data.keys()))

    colors = plot_format.get("colors", [None] * len(v_labels) * len(grp_labels))
    rects_data = nested_defaultdict(list, depth=2)
    x_coord_scale = plot_format.get("scale", {}).get("x", None)
    x_locs = np.arange((len(xlabels)))
    if x_coord_scale:
        x_locs *= x_coord_scale
    for i, stack_label in enumerate(v_labels):
        # calculate coordinates for each category in group
        n_groups = len(fmt_data[stack_label].keys())
        x_increments = np.ones((len(grp_labels),)) * width
        if n_groups % 2 == 0:
            center = int(n_groups / 2)
            width_scaling = np.arange(
                start=1, stop=2 * len(x_increments[center:]) + 1, step=2
            )
            x_increments[center:] = x_increments[center:] * 0.5 * width_scaling
            x_increments[:center] = -x_increments[:center] * 0.5 * width_scaling[::-1]
        else:
            center = int((n_groups - 1) / 2)
            left_width_scale = np.arange(start=1, stop=len(x_increments[:center]) + 1)
            x_increments[:center] = -x_increments[:center] * left_width_scale[::-1]
            x_increments[center] = 0.0
            right_width_scale = np.arange(
                start=1, stop=len(x_increments[center + 1 :]) + 1
            )
            x_increments[center + 1 :] = x_increments[center + 1 :] * right_width_scale
        assert x_increments.shape[0] == len(fmt_data[stack_label].keys())
        for j, grp_label in enumerate(fmt_data[stack_label]):
            series = fmt_data[stack_label][grp_label]
            rects = ax.bar(
                x_locs + x_increments[j],
                series,
                width,
                bottom=start_pos[stack_label][grp_label],
                label=grp_label,
                color=colors[i + j],
            )
            if x_coord_scale:
                ax.set_xticks(x_locs)
                ax.set_xticklabels(xlabels)
            rects_data[stack_label][grp_label] = rects
            if start_pos[stack_label][grp_label] is None:
                start_pos[stack_label][grp_label] = series
            else:
                start_pos[stack_label][grp_label] += series

    ax = format_plot(ax, plot_format)

    return ax, fig


def plot_lines(
    x: Union[list, dict], y: dict, plt_format: dict, skip_keys: set[str] = None
):
    fig, ax = plt.subplots()

    fmt_opts = plt_format.get("fmt", [])
    colors, markers, linestyles = [], [], []
    if not fmt_opts:
        colors = plt_format.get("color", [])
        markers = plt_format.get("marker", [])
        linestyles = plt_format.get("linestyle", [])
        for c, m, ls in zip_longest(colors, markers, linestyles, fillvalue=None):
            if c not in BASE_COLORS:
                if c is None:
                    c = DEFAULT_COLOUR
                else:
                    c = ""
            if ls not in BASE_LINESTYLES:
                if ls is None:
                    ls = DEFAULT_LINESTYLE
                else:
                    ls = ""
            if m is None:
                m = DEFAULT_MARKER

            fmt_opts.append(f"{c}{m}{ls}")
    for idx, (key, fmt) in enumerate(zip_longest(y, fmt_opts, fillvalue=None)):
        if key is None:
            continue
        max_y = max(y[key])
        max_y_idx = y[key].index(max_y)
        color, linestyle = None, None
        if colors:
            color = colors[idx]
        if linestyles:
            linestyle = linestyles[idx]
        if skip_keys:
            if key not in skip_keys:
                if isinstance(x[0], list):
                    print(
                        f"Key: {key}. Max y value of {max_y} attained at {x[idx][max_y_idx]} steps"
                    )
                    ax.plot(
                        x[idx], y[key], fmt, color=color, linestyle=linestyle, label=key
                    )
                else:
                    print(
                        f"Key: {key}. Max y value of {max_y} attained at {x[max_y_idx]} steps"
                    )
                    ax.plot(x, y[key], fmt, color=color, linestyle=linestyle, label=key)
        else:
            if isinstance(x, dict):
                print(
                    f"Key: {key}. Max y value of {max_y} attained at {x[key][max_y_idx]} steps"
                )
                ax.plot(
                    x[key], y[key], fmt, color=color, linestyle=linestyle, label=key
                )
            elif isinstance(x[0], list):
                print(
                    f"Key: {key}. Max y value of {max_y} attained at {x[idx][max_y_idx]} steps"
                )
                ax.plot(
                    x[idx], y[key], fmt, color=color, linestyle=linestyle, label=key
                )
            else:
                print(
                    f"Key: {key}. Max y value of {max_y} attained at {x[max_y_idx]} steps"
                )
                ax.plot(x, y[key], fmt, color=color, linestyle=linestyle, label=key)
    vline_fmt = plt_format.pop("vline", {})
    if vline_fmt:
        x_pos = vline_fmt.pop("x", [])
        for i, x in enumerate(x_pos):
            this_line_kwargs = {
                k: vline_fmt[k][i % len(vline_fmt[k])] for k in vline_fmt
            }
            ax.axvline(x, **this_line_kwargs)
    plt.tight_layout()
    format_plot(ax, plt_format)
