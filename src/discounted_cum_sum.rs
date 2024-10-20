use DataType::Float64;
use num_traits::Zero;
use polars::prelude::*;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct DiscountedCumSumKwargs {
    gamma: f64,
}
 
fn kernel_discounted_cum_sum<T>(inputs: &ChunkedArray<T>, gamma: T::Native) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init: T::Native = T::Native::zero();
    let out: ChunkedArray<T> = inputs
        .iter()
        .scan(init, |running_sum, x| {
            match x {
                Some(x) => {
                    *running_sum = *running_sum * gamma + x;
                    Some(Some(*running_sum))
                },
                None => Some(None)
            }
        })
        .collect_trusted();
    
    out.with_name(inputs.name().clone())
}

#[polars_expr(output_type=Float64)]
pub fn discounted_cum_sum(inputs: &[Series], kwargs: DiscountedCumSumKwargs) -> PolarsResult<Series> {
    polars_ensure!((0.0..=1.0).contains(&kwargs.gamma), ComputeError: "gamma must be in [0, 1]");

    let gamma: f64 = kwargs.gamma;
    let inputs: &Series = &inputs[0];
    let out: Series = match inputs.dtype() {
        Float64 => kernel_discounted_cum_sum(inputs.f64()?, gamma).into_series(),
        _ => {
            let inputs = &inputs.cast(&Float64)?;
            kernel_discounted_cum_sum(inputs.f64()?, gamma).into_series()
        },
    };

    Ok(out)
}