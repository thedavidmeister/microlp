//! [`Basis`]: taken from a pure-LP solution, it starts a later solve of the
//! same model where the earlier one ended.
#[cfg(test)]
mod tests_basis {
    use crate::*;

    fn solution(outcome: SolveOutcome) -> Solution {
        outcome.into_solution().expect("a solution")
    }

    fn unpresolved() -> SolveOptions {
        let mut options = SolveOptions::default();
        options.presolve = false;
        options
    }

    fn from(basis: Basis) -> SolveOptions {
        let mut options = SolveOptions::default();
        options.basis = Some(basis);
        options
    }

    /// max sum of x over a chain x_i + x_{i+1} <= c_i, x in [0, 10]: an
    /// optimum some pivots from the slack basis.
    fn chain(c: &[f64]) -> Problem {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x: Vec<Variable> = (0..=c.len())
            .map(|i| p.add_var(1.0 + i as f64 * 0.1, (0.0, 10.0)))
            .collect();
        for (i, c) in c.iter().enumerate() {
            p.add_constraint([(x[i], 1.0), (x[i + 1], 1.0)], ComparisonOp::Le, *c);
        }
        p
    }

    const C: [f64; 8] = [5.0, 7.0, 4.0, 9.0, 6.0, 8.0, 3.0, 7.0];

    #[test]
    fn a_basis_restarts_at_its_optimum() {
        let p = chain(&C);
        let cold = solution(p.solve_with(unpresolved()).unwrap());
        assert!(cold.stats().lp_iterations > 0);
        let basis = cold.basis().expect("an unpresolved LP has a basis");
        let warm = solution(p.solve_with(from(basis)).unwrap());
        assert_eq!(warm.objective(), cold.objective());
        assert_eq!(warm.stats().lp_iterations, 0);
    }

    #[test]
    fn a_basis_carries_to_moved_right_hand_sides() {
        let first = solution(chain(&C).solve_with(unpresolved()).unwrap());
        let moved = C.map(|c| c + 0.5);
        let p = chain(&moved);
        let cold = solution(p.solve_with(unpresolved()).unwrap());
        let warm = solution(p.solve_with(from(first.basis().unwrap())).unwrap());
        assert!((warm.objective() - cold.objective()).abs() < 1e-9);
        assert!(
            warm.stats().lp_iterations < cold.stats().lp_iterations,
            "warm {} cold {}",
            warm.stats().lp_iterations,
            cold.stats().lp_iterations
        );
    }

    #[test]
    fn statuses_round_trip() {
        let s = solution(chain(&C).solve_with(unpresolved()).unwrap());
        let basis = s.basis().unwrap();
        // a status per variable, then per row, with a basic one per row
        assert_eq!(basis.statuses().len(), 9 + 8);
        let basic = basis
            .statuses()
            .iter()
            .filter(|s| **s == BasisStatus::Basic)
            .count();
        assert_eq!(basic, 8);
        assert_eq!(Basis::from_statuses(basis.statuses().to_vec()), basis);
    }

    #[test]
    fn a_reduced_model_gives_no_basis() {
        // a row no point within the bounds can bind, which presolve drops
        let mut p = chain(&C);
        p.add_constraint(
            [(Variable(0), 1.0), (Variable(1), 1.0)],
            ComparisonOp::Le,
            100.0,
        );
        assert!(solution(p.solve().unwrap()).basis().is_none());
        assert!(solution(p.solve_with(unpresolved()).unwrap())
            .basis()
            .is_some());
    }

    #[test]
    fn a_solve_from_a_basis_keeps_one() {
        // a basis skips presolve, so the model stays the user's
        let mut p = chain(&C);
        p.add_constraint(
            [(Variable(0), 1.0), (Variable(1), 1.0)],
            ComparisonOp::Le,
            100.0,
        );
        let basis = solution(p.solve_with(unpresolved()).unwrap())
            .basis()
            .unwrap();
        assert!(solution(p.solve_with(from(basis)).unwrap())
            .basis()
            .is_some());
    }

    #[test]
    fn an_added_row_joins_the_basis() {
        let s = solution(chain(&C).solve_with(unpresolved()).unwrap());
        let x0 = Variable(0);
        let s = solution(
            s.add_constraint([(x0, 1.0)], ComparisonOp::Le, 1.0)
                .unwrap(),
        );
        let basis = s.basis().unwrap();
        assert_eq!(basis.statuses().len(), 9 + 9);
        let mut p = chain(&C);
        p.add_constraint([(x0, 1.0)], ComparisonOp::Le, 1.0);
        let warm = solution(p.solve_with(from(basis)).unwrap());
        assert!((warm.objective() - s.objective()).abs() < 1e-9);
        assert_eq!(warm.stats().lp_iterations, 0);
    }

    #[test]
    fn a_mip_has_no_basis_and_takes_none() {
        let mut p = chain(&C);
        let z = p.add_integer_var(1.0, (0, 3));
        p.add_constraint([(z, 1.0)], ComparisonOp::Le, 2.5);
        let s = solution(p.solve_with(unpresolved()).unwrap());
        assert!(s.basis().is_none());
        let lp = solution(chain(&C).solve_with(unpresolved()).unwrap());
        match p.solve_with(from(lp.basis().unwrap())) {
            Err(Error::InvalidOptions(m)) => assert!(m.contains("pure LP"), "{m}"),
            other => panic!("expected the basis refused, got {other:?}"),
        }
    }

    #[test]
    fn a_basis_of_the_wrong_shape_is_refused() {
        let p = chain(&C);
        let refused = |statuses: Vec<BasisStatus>, got: &str| match p
            .solve_with(from(Basis::from_statuses(statuses)))
        {
            Err(Error::InvalidOptions(m)) => {
                assert!(m.contains("expected 17 statuses with 8 basic"), "{m}");
                assert!(m.contains(got), "{m}");
            }
            other => panic!("expected the basis refused, got {other:?}"),
        };
        let mut right = vec![BasisStatus::AtLower; 9];
        right.extend([BasisStatus::Basic; 8]);
        refused(right[1..].to_vec(), "got 16 with 8 basic");
        let mut short = right.clone();
        short[16] = BasisStatus::AtUpper;
        refused(short, "got 17 with 7 basic");
        // the slack basis itself is a fit
        assert!(p.solve_with(from(Basis::from_statuses(right))).is_ok());
    }
}
