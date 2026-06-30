//! Small numeric helpers shared by the pipeline finalization filters.

/// Median of `values`, or `None` when empty.
///
/// Sorts with a NaN-tolerant comparator (`partial_cmp` falling back to `Equal`);
/// callers that care about NaN should pre-filter non-finite values.
pub(super) fn median_f64(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_empty_is_none() {
        assert_eq!(median_f64(vec![]), None);
    }

    #[test]
    fn median_odd_is_middle() {
        assert_eq!(median_f64(vec![3.0, 1.0, 2.0]), Some(2.0));
    }

    #[test]
    fn median_even_is_average_of_middles() {
        assert_eq!(median_f64(vec![4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }
}
