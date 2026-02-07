use std::collections::HashMap;

pub(super) fn solve_circle_center_mm(
    points: &[[f64; 2]],
    init_center_mm: [f64; 2],
    radius_mm: f64,
    max_iters: usize,
    huber_delta_mm: f64,
) -> Option<[f64; 2]> {
    if points.is_empty() {
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
        problem.add_residual_block(
            1,
            &["c"],
            Box::new(CircleFactor {
                x: p[0],
                y: p[1],
                radius: radius_mm,
            }),
            Some(Box::new(tiny_solver::loss_functions::HuberLoss::new(
                huber_delta_mm,
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
