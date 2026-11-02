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

/// A dense LP over `n` variables and `m` rows with fixed pseudo-random positive coefficients,
/// minimizing a positive cost over rows of at least a positive amount: infeasible at the origin,
/// so the dual phase starts it.
fn covering(n: usize, m: usize) -> Problem {
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        1.0 + (seed % 97) as f64
    };
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let vars: Vec<Variable> = (0..n)
        .map(|_| p.add_var(next(), (0.0, f64::INFINITY)))
        .collect();
    for _ in 0..m {
        let row: Vec<(Variable, f64)> = vars.iter().map(|v| (*v, next())).collect();
        p.add_constraint(row, ComparisonOp::Ge, 1000.0 * next());
    }
    p
}

/// `p` solved `limit` pivots a call, resumed until optimal: the calls, the pivots over them,
/// and the outcome. Panics past `max_calls`.
fn sliced(p: &Problem, limit: u64, max_calls: u64) -> (u64, u64, SolveOutcome) {
    let mut outcome = p.solve_with(options(Some(limit))).unwrap();
    let mut calls = 1;
    while !outcome.is_optimal() {
        assert!(calls < max_calls, "{calls} calls at {limit} pivots a call");
        outcome = outcome.resume().unwrap();
        calls += 1;
    }
    (calls, outcome.stats().lp_iterations, outcome)
}

#[test]
fn a_sliced_solve_makes_the_pivots_of_the_whole_one() {
    for p in [
        dense(30, 20),
        dense(60, 40),
        covering(30, 20),
        covering(60, 40),
    ] {
        let whole = p.solve_with(options(None)).unwrap();
        let pivots = whole.stats().lp_iterations;
        let whole = whole.into_solution().unwrap();
        for limit in [1, 2, 3, 20] {
            let (calls, spent, outcome) = sliced(&p, limit, 10 * pivots + 10);
            // every call keeps its pivots: the same pivots as the whole solve, `limit` a call
            assert_eq!(spent, pivots, "at {limit} pivots a call");
            assert_eq!(
                calls,
                pivots.div_ceil(limit).max(1),
                "at {limit} pivots a call"
            );
            let sliced = outcome.into_solution().unwrap();
            assert_eq!(
                sliced.objective(),
                whole.objective(),
                "at {limit} pivots a call"
            );
        }
    }
}

#[test]
fn an_interrupted_solve_brackets_the_optimum() {
    let problems = [
        (dense(30, 20), OptimizationDirection::Maximize),
        (dense(60, 40), OptimizationDirection::Maximize),
        (covering(30, 20), OptimizationDirection::Minimize),
        (covering(60, 40), OptimizationDirection::Minimize),
    ];
    for (p, direction) in problems {
        let optimum = p.solve().unwrap().into_solution().unwrap().objective();
        let tol = 1e-9 * optimum.abs().max(1.0);
        // `a` no better than `b` in the problem's direction
        let no_better = |a: f64, b: f64| match direction {
            OptimizationDirection::Maximize => a <= b + tol,
            OptimizationDirection::Minimize => a >= b - tol,
        };
        let mut outcome = p.solve_with(options(Some(1))).unwrap();
        let mut incumbents = 0;
        while let SolveOutcome::Interrupted(mut interrupted) = outcome {
            let bound = interrupted.bound().unwrap();
            assert!(
                no_better(optimum, bound),
                "bound {bound}, optimum {optimum}"
            );
            if let Some(incumbent) = interrupted.incumbent() {
                incumbents += 1;
                assert!(
                    no_better(incumbent.objective(), optimum),
                    "{}",
                    incumbent.objective()
                );
            }
            outcome = SolveOutcome::Interrupted(interrupted).resume().unwrap();
        }
        // a dense LP starts feasible at the origin and stays feasible
        if direction == OptimizationDirection::Maximize {
            assert!(incumbents > 0);
        }
    }
}

#[test]
fn an_infeasible_point_is_no_incumbent() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, 10.0));
    p.add_constraint([(x, 1.0)], ComparisonOp::Ge, 2.0);
    let SolveOutcome::Interrupted(mut at_origin) = p.solve_with(unpresolved(Some(0))).unwrap()
    else {
        panic!("no pivots stops before the first");
    };
    // x = 0 misses x >= 2
    assert!(at_origin.incumbent().is_none());
}

#[test]
fn a_problem_with_integers_has_no_incumbent_or_bound_to_slice() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_integer_var(1.0, (0, 10));
    p.add_constraint([(x, 2.0)], ComparisonOp::Le, 7.0);
    p.set_time_limit(std::time::Duration::ZERO);
    if let SolveOutcome::Interrupted(mut interrupted) = p.solve().unwrap() {
        assert!(interrupted.incumbent().is_none());
        assert!(interrupted.bound().is_none());
    }
}

fn unpresolved(pivot_limit: Option<u64>) -> SolveOptions {
    SolveOptions {
        presolve: false,
        ..options(pivot_limit)
    }
}

#[test]
fn an_incumbent_meets_the_rows_and_is_its_objective() {
    // max 3x + 2y, x + y <= 4, x + 3y <= 6, 0 <= x, y <= 10: vertices (4, 0) at 12, (3, 1) at
    // 11 and (0, 2) at 4, so 12 is the optimum
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_var(3.0, (0.0, 10.0));
    let y = p.add_var(2.0, (0.0, 10.0));
    p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
    p.add_constraint([(x, 1.0), (y, 3.0)], ComparisonOp::Le, 6.0);
    let mut outcome = p.solve_with(unpresolved(Some(0))).unwrap();
    let SolveOutcome::Interrupted(first) = &mut outcome else {
        panic!("no pivots stops before the first");
    };
    // the slack basis's multipliers are zero: each var at its upper bound bounds the optimum
    assert_eq!(first.bound(), Some(3.0 * 10.0 + 2.0 * 10.0));
    while let SolveOutcome::Interrupted(mut interrupted) = outcome {
        assert!(interrupted.bound().unwrap() >= 12.0 - 1e-9);
        if let Some(incumbent) = interrupted.incumbent() {
            let (vx, vy) = (incumbent.var_value(x), incumbent.var_value(y));
            assert!(
                vx + vy <= 4.0 + 1e-9 && vx + 3.0 * vy <= 6.0 + 1e-9,
                "({vx}, {vy})"
            );
            assert!(
                (0.0..=10.0).contains(&vx) && (0.0..=10.0).contains(&vy),
                "({vx}, {vy})"
            );
            assert_eq!(incumbent.objective(), 3.0 * vx + 2.0 * vy);
            assert!(incumbent.objective() <= 12.0 + 1e-9);
        }
        outcome = SolveOutcome::Interrupted(interrupted)
            .resume_with(ResumeOptions {
                pivot_limit: Some(1),
                ..ResumeOptions::default()
            })
            .unwrap();
    }
    assert_eq!(outcome.into_solution().unwrap().objective(), 12.0);
}
