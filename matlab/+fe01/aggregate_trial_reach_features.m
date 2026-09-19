function result = aggregate_trial_reach_features(reachIntervals, tracker, varargin)
%AGGREGATE_TRIAL_REACH_FEATURES Aggregate reach duration and path ratio.
result = fe01.fe01_internal("aggregate_trial_reach_features", reachIntervals, tracker, varargin{:});
end
