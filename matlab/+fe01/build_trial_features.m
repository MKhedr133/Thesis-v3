function result = build_trial_features(trialMetadata, pace, locating, reach, varargin)
%BUILD_TRIAL_FEATURES Build one fixed-schema trial-feature row.
result = fe01.fe01_internal("build_trial_features", trialMetadata, pace, locating, reach, varargin{:});
end
