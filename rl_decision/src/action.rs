//! The action space and the one cost model everything else is built on.
//!
//! The same [`loss`] function prices an action for the risk model's
//! expected-loss estimate, for the sandbox's reward, and for the reward
//! assigned when a real label (human verdict / honeytoken) arrives. One
//! definition keeps simulated and real learning signals consistent.

use std::time::Duration;

use serde::{Deserialize, Serialize};

pub const N_ACTIONS: usize = 4;

/// What the decision layer can ask for. Deliberately small and graded:
/// every variant is something `ips_control_plane` can already execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(usize)]
pub enum Action {
    /// No autonomous containment: route to a human reviewer.
    Escalate = 0,
    /// Short quarantine (15 min): cheap to be wrong about.
    BanShort = 1,
    /// Standard quarantine (2 h) -- the legacy `High` tier.
    BanStandard = 2,
    /// Extended quarantine (24 h) -- the legacy `Critical` tier.
    BanExtended = 3,
}

pub const ALL_ACTIONS: [Action; N_ACTIONS] =
    [Action::Escalate, Action::BanShort, Action::BanStandard, Action::BanExtended];

impl Action {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: usize) -> Option<Action> {
        ALL_ACTIONS.get(i).copied()
    }

    pub fn is_ban(self) -> bool {
        !matches!(self, Action::Escalate)
    }

    pub fn ban_duration(self) -> Option<Duration> {
        match self {
            Action::Escalate => None,
            Action::BanShort => Some(Duration::from_secs(15 * 60)),
            Action::BanStandard => Some(Duration::from_secs(2 * 3600)),
            Action::BanExtended => Some(Duration::from_secs(24 * 3600)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Action::Escalate => "Escalate",
            Action::BanShort => "BanShort",
            Action::BanStandard => "BanStandard",
            Action::BanExtended => "BanExtended",
        }
    }
}

/// Loss if the host really is malicious (`miss`) and if it is benign
/// (`collateral`), per action. Units are arbitrary but consistent: roughly
/// "analyst-minutes-equivalents / 10".
///
/// * `Escalate` leaves the attacker active until a human acts, but costs
///   only analyst time if the host was benign.
/// * Longer bans contain better (lower miss loss) but cost more if wrong.
/// * `severity` (0 = Low .. 3 = Critical) scales the miss loss: failing to
///   stop a Critical attack is worse than failing to stop a Low one.
/// * `protected` hosts (gateway, DNS, the servers being defended) make a
///   wrongful ban several times more expensive.
pub fn loss(action: Action, malicious: bool, severity: u8, protected: bool) -> f64 {
    let sev_scale = 1.0 + 0.5 * f64::from(severity.min(3));
    let (miss, collateral) = match action {
        Action::Escalate => (4.0, 0.5),
        Action::BanShort => (2.5, 1.5),
        Action::BanStandard => (1.2, 5.0),
        Action::BanExtended => (0.6, 18.0),
    };
    if malicious {
        miss * sev_scale
    } else if protected && action.is_ban() {
        collateral * 5.0
    } else {
        collateral
    }
}

/// Reward = negative loss, scaled to roughly [-1, 0].
pub fn reward(action: Action, malicious: bool, severity: u8, protected: bool) -> f64 {
    -loss(action, malicious, severity, protected) / 10.0
}

/// The legacy (pre-RL) behaviour, for shadow-mode regret comparison:
/// `ips_control_plane::ban_duration_for` banned only High/Critical.
pub fn legacy_action(severity: u8) -> Action {
    match severity {
        3 => Action::BanExtended,
        2 => Action::BanStandard,
        _ => Action::Escalate,
    }
}
