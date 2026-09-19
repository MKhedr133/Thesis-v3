function paths = build_trial_paths(row, rawRoot)
%BUILD_TRIAL_PATHS Resolve the exact tracker and participant-detail paths.
paths = fe01.fe01_internal("build_trial_paths", row, rawRoot);
end
