//! [`Basis`]: where a pure-LP solve ended, kept so a later solve of the same
//! model can start there.

pub use crate::solver::VarStatus as BasisStatus;

/// A simplex basis of a pure LP: a [`BasisStatus`] for each variable in the
/// order added, then for each constraint's slack in the order added. With the
/// model's bounds it fixes a vertex, so a solve given it
/// ([`SolveOptions::basis`](crate::SolveOptions::basis)) starts from that
/// vertex instead of from scratch: when the model changed only a little (a
/// bound, a right-hand side), the old optimum is a few pivots from the new.
///
/// Take one from [`Solution::basis`](crate::Solution::basis); keep it across
/// processes with [`Basis::statuses`] and [`Basis::from_statuses`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Basis {
    pub(crate) statuses: Vec<BasisStatus>,
}

impl Basis {
    /// The statuses, variables first, then slacks.
    pub fn statuses(&self) -> &[BasisStatus] {
        &self.statuses
    }

    /// A basis from `statuses` as [`Basis::statuses`] gives them. A solve
    /// refuses one whose shape does not fit its model.
    pub fn from_statuses(statuses: Vec<BasisStatus>) -> Self {
        Self { statuses }
    }
}
