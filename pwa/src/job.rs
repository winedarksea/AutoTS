//! Explicit compute lifecycle state shared by forecast controls and status UI.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JobState {
    #[default]
    Ready,
    Running,
    Cancelling,
    Restarting,
    Failed,
}

impl JobState {
    pub fn from_lifecycle_message(message: &str) -> Self {
        match message {
            "cancelling" => Self::Cancelling,
            "restarting" => Self::Restarting,
            "failed" => Self::Failed,
            _ => Self::Ready,
        }
    }

    pub fn blocks_data_loading(self) -> bool {
        matches!(self, Self::Running | Self::Cancelling | Self::Restarting)
    }

    pub fn is_forecasting(self) -> bool {
        self == Self::Running
    }

    pub fn display_label(self) -> &'static str {
        match self {
            Self::Ready => "Compute ready",
            Self::Running => "Forecast running",
            Self::Cancelling => "Cancelling forecast",
            Self::Restarting => "Restarting compute",
            Self::Failed => "Compute unavailable",
        }
    }

    pub fn status_role_class(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Cancelling | Self::Restarting => "warning",
            Self::Failed => "failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::JobState;

    #[test]
    fn only_active_lifecycle_states_block_data_loading() {
        assert!(JobState::Running.blocks_data_loading());
        assert!(JobState::Cancelling.blocks_data_loading());
        assert!(JobState::Restarting.blocks_data_loading());
        assert!(!JobState::Ready.blocks_data_loading());
        assert!(!JobState::Failed.blocks_data_loading());
    }

    #[test]
    fn states_have_stable_user_facing_labels_and_roles() {
        assert_eq!(JobState::Ready.display_label(), "Compute ready");
        assert_eq!(JobState::Running.display_label(), "Forecast running");
        assert_eq!(JobState::Cancelling.status_role_class(), "warning");
        assert_eq!(JobState::Restarting.status_role_class(), "warning");
        assert_eq!(JobState::Failed.display_label(), "Compute unavailable");
    }
}
