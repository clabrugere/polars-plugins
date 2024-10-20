use pyo3_polars::PolarsAllocator;

mod discounted_cum_sum;
mod feature_hasher;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();