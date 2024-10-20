# polars-plugins

Demonstrates how to implement custom expressions on polars `Series` using their plugins framework. This is super convenient for executing compute intensive operations fast by leveraging the rust runtime and all the higher level optimizations pola-rs provides.

## Getting Started

We implement two custom operations:
- a discounted cumulative sum that basically computes $y_{t+1} = \gamma * y_t + x_t$ with $y_0 = x_0$. This operation is sequential by nature and so cannot really be parallelized (expect if it runs on independent groups).
- a feature hasher that applies the hashing trick to map strings to integers. This is very useful for online and large scale machine learning systems as it's usually a cheap operation, stateless and elegantly handle out of vocabulary modalities and missing values.

As recommended by the polars documentation, repos implementing extensions should have the following structure:

```
├ my_lib # python package binding the rust code
|  └ __init__.py
|
├ src # rust package implementing the logic
|  ├ my_lib_implem.rs
|  └ lib.rs
|
├ Cargo.toml # describes the rust crate (name, rust version, dependencies, ...)
└ pyproject.toml # describes the python package
```

It's important that the package and lib names declared in Cargo.toml are the same than the python package name (here `my_lib`)!

```toml
[package]
name = "polars_plugins" # <- here
version = "0.1.0"
edition = "2021"

[lib]
name = "polars_plugins" # <- and here
crate-type = ["cdylib"]
```

`__init__.py` declares methods that register new polars expressions and returns them. This binds the rust lib to python so that it can be called by the interpreter at runtime. `lib.rs` exposes the rust modules to make the binding work. The binding is handled by `pyo3` under the hood.

```python
def discounted_cum_sum(expr: pl.Expr, gamma: float) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="discounted_cum_sum",
        args=[expr],
        kwargs={"gamma": gamma},
        is_elementwise=False, # <- we can control how this expression should behave on window and aggregations
    )
```

The rust code need to be compiled and we use `maturin` for that by running `maturing build` to build the crate into python package, or `maturing dev --uv` to build and install the crate in the local venv (using uv here).

As far as I explored, developing custom expressions in rust is quite easy. Most of the time you will implement a method with the signature 

```rust
pub fn my_func(inputs: &[Series], kwargs: StructKwargs) -> PolarsResult<Series> {}
``` 

taking by reference multiple Series and optionally a struct storing arguments passed from the python caller and returning a new Series. You will need to decorate it with a macro to specify the type of the output data and to generate a bunch of code at compile time.

```rust
#[polars_expr(output_type=UInt64)]
```

As seen in `discounted_cum_sum.rs` you can handle multiple types quite easily by implementing a generic kernel and simply matching on the input `dtype` in the bound method.

## Usage

Once installed, we only need to import the new expressions and voilà!

```python
import numpy as np
import polars as pl

from polars_plugins import discounted_cum_sum, feature_hasher


print(
    pl.DataFrame({"some_values": np.random.randint(0, 10, size=1000).tolist()}).with_columns(
        cum_sum=pl.all().cum_sum(),
        discounted_cum_sum=discounted_cum_sum(pl.all(), gamma=0.9),
    )
)
```
```
out:
┌─────────────┬─────────┬────────────────────┐
│ some_values ┆ cum_sum ┆ discounted_cum_sum │
│ ---         ┆ ---     ┆ ---                │
│ i64         ┆ i64     ┆ f64                │
╞═════════════╪═════════╪════════════════════╡
│ 0           ┆ 0       ┆ 0.0                │
│ 3           ┆ 3       ┆ 3.0                │
│ 2           ┆ 5       ┆ 4.7                │
│ 5           ┆ 10      ┆ 9.23               │
│ 7           ┆ 17      ┆ 15.307             │
│ 9           ┆ 26      ┆ 22.7763            │
│ 1           ┆ 27      ┆ 21.49867           │
│ 0           ┆ 27      ┆ 19.348803          │
│ 5           ┆ 32      ┆ 22.413923          │
│ 6           ┆ 38      ┆ 26.17253           │
└─────────────┴─────────┴────────────────────┘
```

```python
vocab = ["chrome", "firefox", "safari", "bing", None]
(
    pl.DataFrame({"feature": np.random.choice(vocab, size=1000).tolist()})
    .with_columns(hashed=feature_hasher("feature", num_buckets=11))
)

```

```
out:
┌─────────┬────────┐
│ feature ┆ hashed │
│ ---     ┆ ---    │
│ str     ┆ u64    │
╞═════════╪════════╡
│ bing    ┆ 6      │
│ null    ┆ 0      │
│ bing    ┆ 6      │
│ firefox ┆ 2      │
│ null    ┆ 0      │
│ firefox ┆ 2      │
│ chrome  ┆ 9      │
│ bing    ┆ 6      │
│ firefox ┆ 2      │
│ firefox ┆ 2      │
└─────────┴────────┘
```