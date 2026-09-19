function result = detect_focus_episodes(tracker, catalog)
%DETECT_FOCUS_EPISODES Detect complete engine-labelled focus episodes.
result = fe01.fe01_internal("detect_focus_episodes", tracker, catalog);
end
