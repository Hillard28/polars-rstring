from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from polars._typing import IntoExpr

PLUGIN_PATH = Path(__file__).parent

def distance(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="distance",
        is_elementwise=True,
    )

def normalized_distance(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="normalized_distance",
        is_elementwise=True,
    )

def similarity(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="similarity",
        is_elementwise=True,
    )

def normalized_similarity(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="normalized_similarity",
        is_elementwise=True,
    )

def partial_distance(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="partial_distance",
        is_elementwise=True,
    )

def normalized_partial_distance(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="normalized_partial_distance",
        is_elementwise=True,
    )

def partial_similarity(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="partial_similarity",
        is_elementwise=True,
    )

def normalized_partial_similarity(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        args=[expr, other],
        plugin_path=PLUGIN_PATH,
        function_name="normalized_partial_similarity",
        is_elementwise=True,
    )
