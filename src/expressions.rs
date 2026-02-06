#![allow(clippy::unused_unit)]
use polars::prelude::*;
//use polars_arrow::legacy::prelude::*;
use polars::prelude::arity::binary_elementwise_values;
use pyo3_polars::derive::polars_expr;

// Compute simple Levenshtein distance function

pub fn rdistance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let n = a_chars.len();
    let m = b_chars.len();

    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }

    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];

    for (i, &ac) in a_chars.iter().enumerate() {
        curr[0] = i + 1;
        for j in 0..m {
            let cost = if ac == b_chars[j] { 0 } else { 1 };
            let deletion = prev[j + 1] + 1;
            let insertion = curr[j] + 1;
            let substitution = prev[j] + cost;
            curr[j + 1] = deletion.min(insertion).min(substitution);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// Returns the Levenshtein distance normalized to [0.0, 1.0].
/// 0.0 means identical, 1.0 means completely different (relative to max length).
pub fn rnormalized_distance(a: &str, b: &str) -> f64 {
    let d = rdistance(a, b) as f64;
    let max = a.chars().count().max(b.chars().count()) as f64;
    if max == 0.0 {
        0.0
    } else {
        d / max
    }
}

/// Returns a raw similarity score: max_length - distance.
pub fn rsimilarity(a: &str, b: &str) -> usize {
    let max = a.chars().count().max(b.chars().count());
    max.saturating_sub(rdistance(a, b))
}

/// Returns similarity normalized to [0.0, 1.0].
pub fn rnormalized_similarity(a: &str, b: &str) -> f64 {
    let max = a.chars().count().max(b.chars().count()) as f64;
    if max == 0.0 {
        1.0
    } else {
        1.0 - (rdistance(a, b) as f64 / max)
    }
}

/// Computes the minimal Levenshtein distance between the smaller of the
/// two input strings and any contiguous substring of the larger string
/// with the same character length as the smaller string. This effectively
/// rolls the smaller string along the larger one and returns the best match
/// distance (0 for an exact substring match).
pub fn rpartial_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    // Identify shorter and longer by character count
    let (short_chars, long_chars) = if a_chars.len() <= b_chars.len() {
        (a_chars, b_chars)
    } else {
        (b_chars, a_chars)
    };

    let n = short_chars.len();
    let m = long_chars.len();

    if n == 0 {
        return 0;
    }

    // If lengths equal, just return full distance
    if n == m {
        let short_str: String = short_chars.iter().collect();
        let long_str: String = long_chars.iter().collect();
        return rdistance(&short_str, &long_str);
    }

    let short_str: String = short_chars.iter().collect();
    let mut min_dist: usize = usize::MAX;

    // Slide window of length `n` over the longer string
    for start in 0..=m - n {
        let window: String = long_chars[start..start + n].iter().collect();
        let d = rdistance(&short_str, &window);
        if d < min_dist {
            min_dist = d;
            if min_dist == 0 {
                break;
            }
        }
    }

    min_dist
}

/// Returns the Levenshtein distance normalized to [0.0, 1.0].
/// 0.0 means identical, 1.0 means completely different (relative to max length).
pub fn rnormalized_partial_distance(a: &str, b: &str) -> f64 {
    let d = rpartial_distance(a, b) as f64;
    let min = a.chars().count().min(b.chars().count()) as f64;
    if min == 0.0 {
        0.0
    } else {
        d / min
    }
}

/// Returns a raw similarity score: max_length - distance.
pub fn rpartial_similarity(a: &str, b: &str) -> usize {
    let min = a.chars().count().min(b.chars().count());
    min.saturating_sub(rpartial_distance(a, b))
}

/// Returns similarity normalized to [0.0, 1.0].
pub fn rnormalized_partial_similarity(a: &str, b: &str) -> f64 {
    let min = a.chars().count().min(b.chars().count()) as f64;
    if min == 0.0 {
        1.0
    } else {
        1.0 - (rpartial_distance(a, b) as f64 / min)
    }
}

#[polars_expr(output_type=Float64)]
fn distance(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| rdistance(x, y) as f64
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| rnormalized_distance(x, y)
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| rsimilarity(x, y) as f64
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| rnormalized_similarity(x, y)
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn partial_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| rpartial_distance(x, y) as f64
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_partial_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| rnormalized_partial_distance(x, y)
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn partial_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| rpartial_similarity(x, y) as f64
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normalized_partial_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| rnormalized_partial_similarity(x, y)
    );
    Ok(out.into_series())
}
