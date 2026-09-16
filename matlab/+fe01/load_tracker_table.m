function result = load_tracker_table(path)
%LOAD_TRACKER_TABLE Read one tracker and repair a unique malformed time header.
result = fe01.fe01_internal("load_tracker_table", path);
end
