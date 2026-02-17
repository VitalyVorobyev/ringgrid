/// Golden-section search for the minimum of `f` on `[a, b]`.
///
/// Returns `(x_min, f_min)`.
pub(super) fn golden_section_minimize(
    f: impl Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    max_evals: usize,
) -> (f64, f64) {
    const PHI: f64 = 1.618_033_988_749_895;
    const RESP: f64 = 2.0 - PHI;

    let mut x1 = a + RESP * (b - a);
    let mut x2 = b - RESP * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    let mut evals = 2;

    while evals < max_evals && (b - a).abs() > 1e-18 {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + RESP * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - RESP * (b - a);
            f2 = f(x2);
        }
        evals += 1;
    }

    if f1 < f2 {
        (x1, f1)
    } else {
        (x2, f2)
    }
}
