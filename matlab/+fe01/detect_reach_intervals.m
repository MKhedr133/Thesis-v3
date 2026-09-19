function result = detect_reach_intervals(tracker, grabEvents, varargin)
%DETECT_REACH_INTERVALS Detect final sustained movement blocks before grabs.
result = fe01.fe01_internal("detect_reach_intervals", tracker, grabEvents, varargin{:});
end
