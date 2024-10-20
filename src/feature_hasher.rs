use polars::prelude::*;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use pyo3_polars::derive::polars_expr;
use fastmurmur3::hash;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct FeatureHasherKwargs {
    num_buckets: i64,
}

#[polars_expr(output_type=UInt64)]
pub fn feature_hasher(inputs: &[Series], kwargs: FeatureHasherKwargs) -> PolarsResult<Series> {
    polars_ensure!(kwargs.num_buckets > 1, ComputeError: "num_buckets must be at least 2");

    let num_buckets = kwargs.num_buckets as u128;
    let inputs = inputs[0].str()?;
    let out: ChunkedArray<UInt64Type> = inputs
        .iter()
        .map(|v| {
            match v {
                Some(v) => {
                    let index = (hash(v.as_bytes()) % (num_buckets - 1)) + 1;
                    Some(index as u64)
                },
                _ => Some(0),
            }
        })
        .collect_trusted();

    Ok(out.into_series())
}