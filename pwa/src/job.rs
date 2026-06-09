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
}
