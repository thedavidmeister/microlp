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

/// `dense` with a row its others keep it from meeting: infeasible, and proved so only by
/// pivoting from the origin.
fn unreachable(n: usize, m: usize) -> Problem {
    let mut p = dense(n, m);
    // each row holds the sum of the variables below its right-hand side over its least
    // coefficient, at most 97 * 1000 / 1
    p.add_constraint((0..n).map(|i| (Variable(i), 1.0)), ComparisonOp::Ge, 1e6);
    p
}

/// The error `p` fails with and the pivots it took, solved `first` pivots and resumed `each` a
/// call.
fn failed(p: &Problem, first: u64, each: Option<u64>) -> (Error, u64) {
    let mut outcome = match p.solve_with_stats(unpresolved(Some(first))) {
        Ok(outcome) => outcome,
        Err((e, stats)) => return (e, stats.lp_iterations),
    };
    loop {
        let resume = ResumeOptions {
            pivot_limit: each,
            ..ResumeOptions::default()
        };
        match outcome.resume_with_stats(resume) {
            Ok(next) => outcome = next,
            Err((e, stats)) => return (e, stats.lp_iterations),
        }
    }
}

#[test]
fn a_failed_resume_reports_the_pivots_it_made() {
    let p = unreachable(30, 20);
    let (whole_error, whole) = match p.solve_with_stats(unpresolved(None)) {
        Err((e, stats)) => (e, stats.lp_iterations),
        Ok(_) => panic!("solved an infeasible problem"),
    };
    assert_eq!(whole_error, Error::Infeasible);
    assert!(whole > 0, "{whole} pivots");
    for limit in [1, 2, 3, 20] {
        let (error, pivots) = failed(&p, limit, Some(limit));
        assert_eq!(error, Error::Infeasible, "at {limit} pivots a call");
        assert_eq!(pivots, whole, "at {limit} pivots a call");
    }
}

/// x + y over integers in 0..=5 with 2x + 2y = 3: its relaxation is met at x + y = 1.5, which
/// no integer pair reaches, so only a search proves it infeasible.
fn odd() -> Problem {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_integer_var(1.0, (0, 5));
    let y = p.add_integer_var(1.0, (0, 5));
    p.add_constraint([(x, 2.0), (y, 2.0)], ComparisonOp::Eq, 3.0);
    p
}

#[test]
fn a_failed_search_reports_the_pivots_it_made() {
    let options = SolveOptions {
        presolve: false,
        ..SolveOptions::default()
    };
    let (error, stats) = odd().solve_with_stats(options.clone()).unwrap_err();
    assert_eq!(error, Error::Infeasible);
    // the relaxation leaves the slack basis to meet the row
    assert!(stats.lp_iterations > 0, "{stats:?}");
    assert!(stats.nodes_solved > 0, "{stats:?}");
    // stopped after the root and resumed, the search fails having made the same pivots
    let root = odd()
        .solve_with(SolveOptions {
            node_limit: Some(1),
            ..options
        })
        .unwrap();
    assert!(!root.is_optimal());
    let (resumed_error, resumed) = root
        .resume_with_stats(ResumeOptions::default())
        .unwrap_err();
    assert_eq!(resumed_error, Error::Infeasible);
    assert_eq!(resumed.lp_iterations, stats.lp_iterations, "{resumed:?}");
}

#[test]
fn invalid_resume_options_keep_the_stats_so_far() {
    let p = dense(30, 20);
    let outcome = p.solve_with(options(Some(3))).unwrap();
    assert_eq!(outcome.stats().lp_iterations, 3);
    let invalid = ResumeOptions {
        mip_gap: Some(f64::NAN),
        ..ResumeOptions::default()
    };
    let (error, stats) = outcome.resume_with_stats(invalid).unwrap_err();
    assert!(matches!(error, Error::InvalidOptions(_)), "{error:?}");
    assert_eq!(stats.lp_iterations, 3);
}

#[test]
fn an_interrupted_solve_hands_on_its_basis() {
    // stopped partway, its basis starts a solve of the same model that reaches the same
    // optimum in fewer pivots than one from nothing
    let p = covering(30, 20);
    let cold = p.solve_with(unpresolved(None)).unwrap();
    let total = cold.stats().lp_iterations;
    let half = total / 2;
    let going = p.solve_with(unpresolved(Some(half))).unwrap();
    let SolveOutcome::Interrupted(going) = going else {
        panic!("a solve longer than {half} pivots")
    };
    let basis = going.basis().expect("a pure LP keeps a basis");
    let warm = p
        .solve_with(SolveOptions {
            basis: Some(basis),
            ..unpresolved(None)
        })
        .unwrap();
    let (cold, warm_pivots) = (cold.into_solution().unwrap(), warm.stats().lp_iterations);
    let warm = warm.into_solution().unwrap();
    assert!((warm.objective() - cold.objective()).abs() < 1e-6 * cold.objective().abs());
    assert!(warm_pivots < total, "{warm_pivots} of {total}");
}

#[test]
fn a_solve_from_a_moved_basis_has_a_finite_bound_where_every_var_is_bounded() {
    // the dense LP with every var at most 100, and the first held at least `least`: solved from
    // the optimal basis of `least` 0 with the first held at 5, the basis meets neither its rows
    // nor its bounds, and every multiplier the solve stands at prices each var's term at a finite
    // bound; a row's slack, unbounded above, gives a finite term at the multipliers that zero its
    // reduced cost, so no step's bound is infinite
    let boxed = |least: f64| {
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            1.0 + (seed % 97) as f64
        };
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let vars: Vec<Variable> = (0..40)
            .map(|i| p.add_var(next(), (if i == 0 { least } else { 0.0 }, 100.0)))
            .collect();
        for _ in 0..30 {
            let row: Vec<(Variable, f64)> = vars.iter().map(|v| (*v, next())).collect();
            p.add_constraint(row, ComparisonOp::Le, 1000.0 * next());
        }
        p
    };
    let cold = boxed(0.0)
        .solve_with(unpresolved(None))
        .unwrap()
        .into_solution()
        .unwrap();
    let moved = boxed(5.0);
    let optimum = moved
        .solve_with(unpresolved(None))
        .unwrap()
        .into_solution()
        .unwrap()
        .objective();
    let mut outcome = moved
        .solve_with(SolveOptions {
            basis: cold.basis(),
            ..unpresolved(Some(1))
        })
        .unwrap();
    let mut steps = 0;
    while let SolveOutcome::Interrupted(mut going) = outcome {
        let bound = going.bound().unwrap();
        assert!(bound.is_finite(), "step {steps}: bound {bound}");
        assert!(bound >= optimum - 1e-9 * optimum.abs(), "{bound} {optimum}");
        outcome = SolveOutcome::Interrupted(going).resume().unwrap();
        steps += 1;
    }
    assert!(steps > 0, "a moved basis takes pivots");
}
