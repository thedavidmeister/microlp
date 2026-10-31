//! The `clock` feature: without it, solves never read a clock and a time
//! limit is refused rather than silently ignored.
#[cfg(test)]
mod tests_clock {
    use crate::*;
    use core::time::Duration;

    /// max x + y with x + y <= 4: optimum 4.
    fn problem() -> (Problem, Variable) {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(1.0, (0.0, 3.0));
        let y = problem.add_var(1.0, (0.0, 3.0));
        problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
        (problem, x)
    }

    fn limited() -> SolveOptions {
        let mut options = SolveOptions::default();
        options.time_limit = Some(Duration::from_secs(10));
        options
    }

    #[test]
    fn a_solve_without_a_limit_needs_no_clock() {
        let (problem, x) = problem();
        let s = problem.solve().unwrap().into_solution().unwrap();
        assert_eq!(s.objective(), 4.0);
        // the edits a solution takes need none either
        let s = s.fix_var(x, 0.5).unwrap().into_solution().unwrap();
        assert_eq!(s.objective(), 3.5);
    }

    #[cfg(not(feature = "clock"))]
    #[test]
    fn without_the_clock_a_time_limit_is_refused() {
        let (mut problem, _) = problem();
        let refused = |r: Result<SolveOutcome, Error>, field: &str| match r {
            Err(Error::InvalidOptions(m)) => {
                assert!(m.contains(field) && m.contains("`clock` feature"), "{m}")
            }
            other => panic!("expected the limit refused, got {other:?}"),
        };
        refused(problem.solve_with(limited()), "SolveOptions.time_limit");
        let mut resume = ResumeOptions::default();
        resume.time_limit = Some(Duration::from_secs(10));
        let outcome = problem.solve().unwrap();
        refused(outcome.resume_with(resume), "ResumeOptions.time_limit");
        problem.set_time_limit(Duration::from_secs(10));
        refused(problem.solve(), "SolveOptions.time_limit");
    }

    #[cfg(not(feature = "clock"))]
    #[test]
    fn without_the_clock_no_time_passes() {
        let (problem, _) = problem();
        let s = problem.solve().unwrap().into_solution().unwrap();
        assert_eq!(s.stats().elapsed, Duration::ZERO);
    }

    #[cfg(feature = "clock")]
    #[test]
    fn with_the_clock_a_time_limit_is_kept() {
        let (problem, _) = problem();
        let s = problem
            .solve_with(limited())
            .unwrap()
            .into_solution()
            .unwrap();
        assert_eq!(s.objective(), 4.0);
    }
}
