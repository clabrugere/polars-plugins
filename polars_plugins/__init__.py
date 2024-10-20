from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function


def discounted_cum_sum(expr: pl.Expr, gamma: float) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="discounted_cum_sum",
        args=[expr],
        kwargs={"gamma": gamma},
        is_elementwise=False,
    )


def feature_hasher(expr: pl.Expr, num_buckets: int) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="feature_hasher",
        args=[expr],
        kwargs={"num_buckets": num_buckets},
        is_elementwise=True,
    )
