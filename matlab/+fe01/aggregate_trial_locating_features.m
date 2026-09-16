function result = aggregate_trial_locating_features(searchIntervals, focusEpisodes, tracker, catalog)
%AGGREGATE_TRIAL_LOCATING_FEATURES Aggregate locating, irrelevant focus, and head turning.
result = fe01.fe01_internal("aggregate_trial_locating_features", searchIntervals, focusEpisodes, tracker, catalog);
end
