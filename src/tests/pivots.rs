//! A pivot budget: a pure LP solved a few pivots at a time, resumed to the same optimum.

use crate::*;

/// A dense LP over `n` variables and `m` rows with fixed pseudo-random positive coefficients,
/// maximizing a positive objective: bounded and feasible at the origin, and many pivots from it.
fn dense(n: usize, m: usize) -> Problem {
    let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        1.0 + (seed % 97) as f64
    };
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let vars: Vec<Variable> = (0..n)
        .map(|_| p.add_var(next(), (0.0, f64::INFINITY)))
        .collect();
    for _ in 0..m {
        let row: Vec<(Variable, f64)> = vars.iter().map(|v| (*v, next())).collect();
        p.add_constraint(row, ComparisonOp::Le, 1000.0 * next());
    }
    p
}

fn options(pivot_limit: Option<u64>) -> SolveOptions {
    SolveOptions {
        pivot_limit,
        ..SolveOptions::default()
    }
}

#[test]
fn a_pivot_budget_resumes_to_the_same_optimum() {
    let p = dense(30, 20);
    let whole = p.solve_with(options(None)).unwrap();
    let total = whole.stats().lp_iterations;
    assert!(total > 6, "{total} pivots");
    let mut outcome = p.solve_with(options(Some(3))).unwrap();
    let mut calls = 1;
    let mut before = 0;
    while !outcome.is_optimal() {
        // each call stops at its budget, between pivots
        assert_eq!(outcome.termination_reason(), TerminationReason::PivotLimit);
        let spent = outcome.stats().lp_iterations - before;
        assert!(spent <= 3, "{spent} pivots in one call");
        before = outcome.stats().lp_iterations;
        outcome = outcome.resume().unwrap();
        calls += 1;
        // each call but the last pivots: the budget is fresh from where the last stopped
        assert!(calls <= total + 1, "{calls} calls for {total} pivots");
    }
    assert!(calls > 2, "{calls} calls");
    let sliced = outcome.into_solution().unwrap();
    let whole = whole.into_solution().unwrap();
    assert!((sliced.objective() - whole.objective()).abs() < 1e-6 * whole.objective().abs());
}

#[test]
fn no_pivots_stops_before_the_first() {
    let p = dense(10, 8);
    let outcome = p.solve_with(options(Some(0))).unwrap();
    assert_eq!(outcome.termination_reason(), TerminationReason::PivotLimit);
    assert_eq!(outcome.stats().lp_iterations, 0);
    // a fresh budget each call: resumed unlimited, it finishes
    let resume = ResumeOptions {
        pivot_limit: None,
        ..ResumeOptions::default()
    };
    assert!(outcome.resume_with(resume).unwrap().is_optimal());
}

#[test]
fn an_optimal_basis_needs_no_pivots_to_finish() {
    let p = dense(10, 8);
    // presolve off, so the basis names the model's own rows
    let cold = p
        .solve_with(SolveOptions {
            presolve: false,
            ..options(None)
        })
        .unwrap()
        .into_solution()
        .unwrap();
    let again = p
        .solve_with(SolveOptions {
            presolve: false,
            basis: cold.basis(),
            ..options(Some(0))
        })
        .unwrap();
    assert!(again.is_optimal());
    assert_eq!(again.stats().lp_iterations, 0);
}

#[test]
fn a_problem_with_integers_refuses_a_pivot_budget() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_integer_var(1.0, (0, 10));
    p.add_constraint([(x, 2.0)], ComparisonOp::Le, 7.0);
    assert!(matches!(
        p.solve_with(options(Some(5))),
        Err(Error::InvalidOptions(_))
    ));
    let outcome = p.solve_with(options(None)).unwrap();
    let resume = ResumeOptions {
        pivot_limit: Some(5),
        ..ResumeOptions::default()
    };
    // an optimal outcome returns as it is, unresumed
    assert!(outcome.resume_with(resume).is_ok());
}
