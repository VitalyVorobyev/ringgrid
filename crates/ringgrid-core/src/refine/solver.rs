use std::collections::HashMap;

use nalgebra::{Matrix2, Vector2};

use super::CircleCenterSolver;

#[inline]
fn huber_weight(abs_r: f64, delta: f64) -> f64 {
    if abs_r <= delta {
        1.0
    } else {
        delta / abs_r
    }
}

pub(super) fn solve_circle_center_mm(
    points: &[[f64; 2]],
    init_center_mm: [f64; 2],
    radius_mm: f64,
    max_iters: usize,
    huber_delta_mm: f64,
    solver: CircleCenterSolver,
) -> Option<[f64; 2]> {
    match solver {
        CircleCenterSolver::Irls => solve_circle_center_mm_irls(
            points,
            init_center_mm,
            radius_mm,
            max_iters,
            huber_delta_mm,
        ),
        CircleCenterSolver::Lm => {
            solve_circle_center_mm_lm(points, init_center_mm, radius_mm, max_iters, huber_delta_mm)
        }
    }
}

fn solve_circle_center_mm_irls(
    points: &[[f64; 2]],
    init_center_mm: [f64; 2],
    radius_mm: f64,
    max_iters: usize,
    huber_delta_mm: f64,
) -> Option<[f64; 2]> {
    if points.len() < 3 {
        return None;
    }
    if !radius_mm.is_finite() || radius_mm <= 0.0 {
        return None;
    }

    let mut center = init_center_mm;
    let iters = max_iters.clamp(1, 80);
    let delta = huber_delta_mm.max(1e-6);

    for _ in 0..iters {
        let mut h = Matrix2::<f64>::zeros();
        let mut g = Vector2::<f64>::zeros();
        let mut n_used = 0usize;

        for p in points {
            let x = p[0];
            let y = p[1];
            if !x.is_finite() || !y.is_finite() {
                continue;
            }

            // Geometric residual with fixed radius:
            //   r_i(c) = ||p_i - c|| - R
            let dx = center[0] - x;
            let dy = center[1] - y;
            let dist = (dx * dx + dy * dy).sqrt();
            if !dist.is_finite() || dist <= 1e-12 {
                continue;
            }
            let r = dist - radius_mm;
            let w = huber_weight(r.abs(), delta);
            if !w.is_finite() || w <= 0.0 {
                continue;
            }

            // Jacobian wrt center components.
            let j = Vector2::new(dx / dist, dy / dist);
            h += w * (j * j.transpose());
            g += w * (j * r);
            n_used += 1;
        }

        if n_used < 2 {
            return None;
        }

        // Light damping for near-degenerate arcs.
        h += Matrix2::<f64>::identity() * 1e-9;

        let step = h.lu().solve(&(-g))?;
        if !step[0].is_finite() || !step[1].is_finite() {
            return None;
        }

        center[0] += step[0];
        center[1] += step[1];

        if (step[0] * step[0] + step[1] * step[1]).sqrt() < 1e-9 {
            break;
        }
    }

    Some(center)
}

fn solve_circle_center_mm_lm(
    points: &[[f64; 2]],
    init_center_mm: [f64; 2],
    radius_mm: f64,
    max_iters: usize,
    huber_delta_mm: f64,
) -> Option<[f64; 2]> {
    if points.len() < 3 {
        return None;
    }
    if !radius_mm.is_finite() || radius_mm <= 0.0 {
        return None;
    }

    use tiny_solver::factors::na as ts_na;
    use tiny_solver::Optimizer;

    #[derive(Debug, Clone)]
    struct CircleFactor {
        x: f64,
        y: f64,
        radius: f64,
    }

    impl<T: ts_na::RealField> tiny_solver::factors::Factor<T> for CircleFactor {
        fn residual_func(&self, params: &[ts_na::DVector<T>]) -> ts_na::DVector<T> {
            let c = &params[0];
            let cx = c[0].clone();
            let cy = c[1].clone();
            let dx = T::from_f64(self.x).unwrap() - cx;
            let dy = T::from_f64(self.y).unwrap() - cy;
            let dist = (dx.clone() * dx + dy.clone() * dy).sqrt();
            let r = dist - T::from_f64(self.radius).unwrap();
            ts_na::DVector::<T>::from_vec(vec![r])
        }
    }

    let mut problem = tiny_solver::Problem::new();
    for p in points {
        let x = p[0];
        let y = p[1];
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        problem.add_residual_block(
            1,
            &["c"],
            Box::new(CircleFactor {
                x,
                y,
                radius: radius_mm,
            }),
            Some(Box::new(tiny_solver::loss_functions::HuberLoss::new(
                huber_delta_mm.max(1e-6),
            ))),
        );
    }

    let mut initial_values = HashMap::<String, ts_na::DVector<f64>>::new();
    initial_values.insert(
        "c".to_string(),
        ts_na::DVector::<f64>::from_vec(vec![init_center_mm[0], init_center_mm[1]]),
    );

    let optimizer = tiny_solver::LevenbergMarquardtOptimizer::default();
    let options = tiny_solver::OptimizerOptions {
        max_iteration: max_iters.clamp(1, 200),
        verbosity_level: 0,
        ..Default::default()
    };
    let result = optimizer.optimize(&problem, &initial_values, Some(options))?;
    let c = result.get("c")?;
    if c.len() != 2 {
        return None;
    }
    let cx = c[0];
    let cy = c[1];
    if !cx.is_finite() || !cy.is_finite() {
        return None;
    }
    Some([cx, cy])
}
