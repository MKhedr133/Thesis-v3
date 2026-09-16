function varargout = fe01_internal(action, varargin)
%FE01_INTERNAL Shared implementation for the native MATLAB FE-01 package.

action = string(action);
switch action
    case "read_table"
        varargout{1} = localReadTable(varargin{1});
    case "standardize_trial_table"
        varargout{1} = localStandardizeTrialTable(varargin{1});
    case "parse_correct_lists"
        varargout{1} = localParseCorrectLists(varargin{1});
    case "canonicalize_product"
        varargout{1} = localCanonicalizeProduct(varargin{1}, varargin{2});
    case "is_catalog_product"
        varargout{1} = localIsCatalogProduct(varargin{1}, varargin{2});
    case "correct_products"
        varargout{1} = localCorrectProducts(varargin{1}, varargin{2}, varargin{3});
    case "build_trial_paths"
        varargout{1} = localBuildTrialPaths(varargin{1}, varargin{2});
    case "read_participant_hand"
        varargout{1} = localReadParticipantHand(varargin{1});
    case "load_tracker_table"
        varargout{1} = localLoadTrackerTable(varargin{1});
    case "prepare_hand_motion"
        varargout{1} = localPrepareHandMotion(varargin{1}, varargin{2});
    case "prepare_headset_rotations"
        varargout{1} = localPrepareHeadsetRotations(varargin{1});
    case "prepare_headset_quaternions"
        varargout{1} = localPrepareHeadsetRotations(varargin{1});
    case "prepare_headset_quaternion_path"
        varargout{1} = localPrepareHeadsetRotations(varargin{1});
    case "detect_product_grab_events"
        varargout{1} = localDetectProductGrabEvents(varargin{1}, varargin{2}, varargin{3}, varargin{4});
    case "detect_focus_episodes"
        varargout{1} = localDetectFocusEpisodes(varargin{1}, varargin{2});
    case "detect_list_visits"
        varargout{1} = localDetectListVisits(varargin{1});
    case "detect_search_intervals"
        varargout{1} = localDetectSearchIntervals(varargin{1}, varargin{2}, varargin{3}, varargin{4});
    case "detect_reach_intervals"
        varargout{1} = localDetectReachIntervals(varargin{:});
    case "aggregate_trial_pace_and_list_rechecks"
        varargout{1} = localAggregatePace(varargin{1}, varargin{2});
    case "aggregate_trial_locating_features"
        varargout{1} = localAggregateLocating(varargin{1}, varargin{2}, varargin{3}, varargin{4});
    case "aggregate_trial_reach_features"
        varargout{1} = localAggregateReach(varargin{:});
    case "derive_error_outcomes"
        varargout{1} = localDeriveErrorOutcomes(varargin{1});
    case "build_product_grab_features"
        varargout{1} = localBuildProductGrabFeatures(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5});
    case "write_product_grab_features"
        localWriteFixedTable(varargin{1}, varargin{2}, localProductGrabColumns());
    case "build_trial_features"
        varargout{1} = localBuildTrialFeatures(varargin{:});
    case "write_trial_features"
        localWriteFixedTable(varargin{1}, varargin{2}, localTrialFeatureColumns());
    case "write_descriptive_feature_plots"
        varargout{1} = localWriteDescriptivePlots(varargin{1}, varargin{2});
    case "run_full_extraction"
        varargout{1} = localRunFullExtraction(varargin{:});
    case "product_grab_columns"
        varargout{1} = localProductGrabColumns();
    case "trial_feature_columns"
        varargout{1} = localTrialFeatureColumns();
    case "qc_columns"
        varargout{1} = localQcColumns();
    otherwise
        error("fe01:UnknownAction", "Unknown FE-01 internal action: %s", action);
end

function tableData = localReadTable(path)
path = string(path);
if ~isfile(path)
    error("fe01:MissingInput", "Input table not found: %s", path);
end
raw = fileread(path);
firstLine = regexp(raw, "^([^\r\n]*)", "tokens", "once");
if isempty(firstLine)
    header = "";
else
    header = string(firstLine{1});
end
if contains(header, ";")
    delimiter = ";";
elseif contains(header, sprintf("\t"))
    delimiter = sprintf("\t");
else
    delimiter = ",";
end
tableData = readtable(path, "Delimiter", delimiter, "TextType", "string", ...
    "VariableNamingRule", "preserve");
end

function tableData = localStandardizeTrialTable(inputData)
if ~istable(inputData)
    error("fe01:InvalidInput", "The master trial input must be a table.");
end
tableData = inputData;
aliases = containers.Map( ...
    {"participant", "participant_id", "session", "session_id", "trial", ...
    "source_tracker_csv_filename", "group", "participant_group", "condition", ...
    "condition_name", "level3", "level", "difficulty", "difficulty_level", ...
    "trial2", "trial_order", "language"}, ...
    {"participant_id", "participant_id", "session_id", "session_id", ...
    "source_tracker_csv_filename", "source_tracker_csv_filename", ...
    "participant_group", "participant_group", "condition_name", "condition_name", ...
    "difficulty_level", "difficulty_level", "difficulty_level", "difficulty_level", ...
    "trial_order", "trial_order", "language"});
names = string(tableData.Properties.VariableNames);
for index = 1:numel(names)
    key = lower(strtrim(names(index)));
    if isKey(aliases, char(key))
        tableData.Properties.VariableNames{index} = aliases(char(key));
    end
end
required = ["participant_id", "session_id", "source_tracker_csv_filename", ...
    "participant_group", "condition_name", "difficulty_level", "trial_order"];
names = string(tableData.Properties.VariableNames);
missing = required(~ismember(required, names));
if ~isempty(missing)
    error("fe01:MissingColumns", "Master trial table is missing required columns: %s", ...
        strjoin(missing, ", "));
end
tableData.participant_id = arrayfun(@localNormalizeParticipant, ...
    tableData.participant_id, "UniformOutput", false);
tableData.participant_id = string(tableData.participant_id);
tableData.session_id = upper(strtrim(string(tableData.session_id)));
tableData.source_tracker_csv_filename = arrayfun(@localTrackerFilename, ...
    tableData.source_tracker_csv_filename, "UniformOutput", false);
tableData.source_tracker_csv_filename = string(tableData.source_tracker_csv_filename);
tableData.participant_group = strtrim(string(tableData.participant_group));
tableData.condition_name = arrayfun(@localNormalizeCondition, ...
    tableData.condition_name, "UniformOutput", false);
tableData.condition_name = string(tableData.condition_name);
tableData.difficulty_level = arrayfun(@localNormalizeDifficulty, ...
    tableData.difficulty_level);
tableData.trial_order = arrayfun(@localNormalizeTrialOrder, ...
    tableData.trial_order, "UniformOutput", false);
tableData.trial_order = string(tableData.trial_order);
if ~ismember("language", string(tableData.Properties.VariableNames))
    tableData.language = repmat("EN", height(tableData), 1);
else
    tableData.language = upper(strtrim(string(tableData.language)));
end
end

function value = localNormalizeParticipant(raw)
tokens = regexp(char(string(raw)), "(\d+)", "tokens", "once");
if isempty(tokens)
    error("fe01:InvalidParticipant", "Participant identifier is not understood: %s", string(raw));
end
value = sprintf("P%02d", str2double(tokens{1}));
end

function value = localNormalizeCondition(raw)
key = lower(strtrim(string(raw)));
switch key
    case "visual"
        value = "Visual";
    case "auditory"
        value = "Auditory";
    case "cognitive"
        value = "Cognitive";
    otherwise
        error("fe01:InvalidCondition", "Condition is not understood: %s", key);
end
end

function value = localNormalizeDifficulty(raw)
value = localNumericScalar(raw);
if ~isfinite(value)
    error("fe01:InvalidDifficulty", "Difficulty is not finite: %s", string(raw));
end
if value == fix(value)
    value = fix(value);
end
end

function value = localNormalizeTrialOrder(raw)
tokens = regexp(char(string(raw)), "(\d+)", "tokens", "once");
if isempty(tokens)
    error("fe01:InvalidTrialOrder", "Trial order is not understood: %s", string(raw));
end
value = sprintf("T%d", str2double(tokens{1}));
end

function value = localTrackerFilename(raw)
text = strtrim(string(raw));
if endsWith(lower(text), ".csv")
    [~, name, extension] = fileparts(text);
    value = name + extension;
    return;
end
tokens = regexp(char(text), "T?(\d+)", "tokens", "once", "ignorecase");
if isempty(tokens)
    error("fe01:InvalidTrackerToken", "Tracker trial token is not understood: %s", text);
end
value = sprintf("data_collector_vr_sample_T%03d.csv", str2double(tokens{1}));
end

function numeric = localNumericVector(values)
if isnumeric(values) || islogical(values)
    numeric = double(values);
else
    numeric = str2double(string(values));
end
numeric = numeric(:);
end

function numeric = localNumericScalar(value)
converted = localNumericVector(value);
if isempty(converted)
    numeric = NaN;
else
    numeric = converted(1);
end
end

function value = localNormalizeHand(raw)
key = lower(strtrim(string(raw)));
if ismember(key, ["r", "right", "right-handed", "right handed"])
    value = "right";
elseif ismember(key, ["l", "left", "left-handed", "left handed"])
    value = "left";
else
    value = missing;
end
end

function paths = localBuildTrialPaths(row, rawRoot)
participant = string(localRecordValue(row, "participant_id"));
if ismissing(participant) || strlength(participant) == 0
    participant = string(localNormalizeParticipant(localRecordValue(row, "participant")));
end
participantNumber = str2double(extractAfter(participant, 1));
session = upper(strtrim(string(localRecordValue(row, "session_id"))));
if ismissing(session) || strlength(session) == 0
    session = upper(strtrim(string(localRecordValue(row, "session"))));
end
filename = string(localRecordValue(row, "source_tracker_csv_filename"));
if ismissing(filename) || strlength(filename) == 0
    filename = string(localTrackerFilename(localRecordValue(row, "trial")));
end
trialRoot = fullfile(string(rawRoot), sprintf("P%d", participantNumber), ...
    participant + "_HMD_Data", participant, session);
paths = struct( ...
    "tracker", fullfile(trialRoot, "trackers", filename), ...
    "participant_details", fullfile(trialRoot, "session_info", "participant_details.csv"));
end

function value = localRecordValue(row, field)
if istable(row)
    if ~ismember(string(field), string(row.Properties.VariableNames))
        value = missing;
        return;
    end
    value = row.(field);
elseif isstruct(row)
    if isfield(row, field)
        value = row.(field);
    else
        value = missing;
    end
else
    value = missing;
end

if iscell(value)
    if isempty(value)
        value = missing;
    else
        value = value{1};
    end
elseif numel(value) > 1
    value = value(1);
end
end

function catalog = localParseCorrectLists(path)
path = string(path);
if ~isfile(path)
    error("fe01:MissingCorrectLists", "Correct-product list not found: %s", path);
end
text = fileread(path);
matches = regexp(text, ...
    "(?s)correctLists\.level(\d+)\.(EN|NL)\s*=\s*\{(.*?)\};", ...
    "tokens", "ignorecase");
ordered = containers.Map("KeyType", "char", "ValueType", "any");
for index = 1:numel(matches)
    token = matches{index};
    difficulty = str2double(token{1});
    language = upper(string(token{2}));
    products = regexp(token{3}, "'([^']+)'", "tokens");
    values = strings(1, numel(products));
    for productIndex = 1:numel(products)
        values(productIndex) = string(products{productIndex}{1});
    end
    ordered(localCatalogKey(difficulty, language)) = values;
end
required = strings(0, 1);
for difficulty = [0, 2, 6, 10]
    required(end+1) = localCatalogKey(difficulty, "EN"); %#ok<AGROW>
    required(end+1) = localCatalogKey(difficulty, "NL"); %#ok<AGROW>
end
missingKeys = required(~arrayfun(@(key) isKey(ordered, char(key)), required));
if ~isempty(missingKeys)
    error("fe01:MissingCorrectLists", "Correct-product definitions are missing: %s", ...
        strjoin(missingKeys, ", "));
end
aliases = containers.Map("KeyType", "char", "ValueType", "char");
allProducts = strings(0, 1);
for difficulty = [0, 2, 6, 10]
    english = ordered(localCatalogKey(difficulty, "EN"));
    dutch = ordered(localCatalogKey(difficulty, "NL"));
    for index = 1:numel(english)
        canonical = localCleanProductName(english(index));
        if strlength(canonical) == 0
            continue;
        end
        aliases(char(canonical)) = char(canonical);
        allProducts(end+1) = canonical; %#ok<AGROW>
        if index <= numel(dutch)
            aliases(char(localCleanProductName(dutch(index)))) = char(canonical);
        end
    end
end
catalog = struct("ordered_lists", ordered, "alias_to_canonical", aliases, ...
    "all_canonical_products", unique(allProducts, "stable"));
end

function key = localCatalogKey(difficulty, language)
key = sprintf("D%d_%s", double(difficulty), upper(string(language)));
end

function value = localCleanProductName(raw)
text = string(raw);
if any(ismissing(text))
    value = "";
    return;
end
value = lower(strtrim(text));
value = regexprep(value, "\s*\(clone\)\s*$", "", "ignorecase");
previous = "__different__";
while value ~= previous
    previous = value;
    value = regexprep(value, "\s*\(\s*\d+\s*\)(?:\s+\d+)*\s*$", "");
    value = regexprep(value, "\s+\d+\s*$", "");
end
value = regexprep(value, "\s+", " ");
value = strtrim(value);
end

function canonical = localCanonicalizeProduct(catalog, raw)
cleaned = localCleanProductName(raw);
if strlength(cleaned) == 0
    canonical = "";
    return;
end
if isfield(catalog, "alias_to_canonical") && isa(catalog.alias_to_canonical, "containers.Map") ...
        && isKey(catalog.alias_to_canonical, char(cleaned))
    canonical = string(catalog.alias_to_canonical(char(cleaned)));
else
    canonical = cleaned;
end
end

function tf = localIsCatalogProduct(catalog, raw)
canonical = localCanonicalizeProduct(catalog, raw);
tf = any(string(catalog.all_canonical_products) == canonical);
end

function products = localCorrectProducts(catalog, difficulty, language)
key = localCatalogKey(difficulty, language);
if isfield(catalog, "ordered_lists") && isa(catalog.ordered_lists, "containers.Map") ...
        && isKey(catalog.ordered_lists, key)
    values = catalog.ordered_lists(key);
    products = strings(1, numel(values));
    for index = 1:numel(values)
        products(index) = localCanonicalizeProduct(catalog, values(index));
    end
else
    products = strings(1, 0);
end
end

function result = localReadParticipantHand(path)
path = string(path);
if ~isfile(path)
    result = struct("hand", missing, "warnings", "participant_details_file_missing");
    return;
end
try
    details = localReadTable(path);
catch exception
    result = struct("hand", missing, ...
        "warnings", "participant_details_read_error:" + string(exception.identifier));
    return;
end
names = lower(strtrim(string(details.Properties.VariableNames)));
candidates = ["side", "dominant_hand", "dominant hand", "handedness", "active_hand"];
index = find(ismember(names, candidates), 1, "first");
if isempty(index)
    result = struct("hand", missing, "warnings", "participant_hand_not_found");
    return;
end
values = strtrim(string(details{:, index}));
values = values(strlength(values) > 0 & ~ismissing(values));
if isempty(values)
    result = struct("hand", missing, "warnings", "participant_hand_not_found");
    return;
end
hand = localNormalizeHand(values(1));
if ismissing(hand)
    result = struct("hand", missing, "warnings", "participant_hand_not_understood");
else
    result = struct("hand", hand, "warnings", strings(0, 1));
end
end

function result = localLoadTrackerTable(path)
tableData = localReadTable(path);
warnings = strings(0, 1);
repaired = false;
names = string(tableData.Properties.VariableNames);
if ~ismember("time", names)
    candidateMask = ~cellfun(@isempty, regexp(cellstr(names), "^\d+time$", "once", "ignorecase"));
    candidates = names(candidateMask);
    if numel(candidates) == 1
        tableData.Properties.VariableNames{strcmp(names, candidates(1))} = "time";
        warnings(end+1) = "repaired_time_header:" + candidates(1); %#ok<AGROW>
        repaired = true;
    else
        error("fe01:MissingTime", "Tracker table is missing required column: time");
    end
end
tableData.time = localNumericVector(tableData.time);
if ~any(isfinite(tableData.time))
    error("fe01:MissingTime", "Tracker table has no usable recorded time values");
end
supportingColumns = ["enableBlackout", "focus_object_name", "focus_object_tag", ...
    "is_left_eye_blinking", "is_right_eye_blinking", "num_items_in_cart", ...
    "is_grabbing_right", "grabbed_object_right", "is_grabbing_left", ...
    "grabbed_object_left", "right_pos_x", "right_pos_y", "right_pos_z", ...
    "left_pos_x", "left_pos_y", "left_pos_z", "hmd_rot_x", "hmd_rot_y", ...
    "hmd_rot_z", "hmd_rot_w"];
numericColumns = ["num_items_in_cart", "right_pos_x", "right_pos_y", ...
    "right_pos_z", "left_pos_x", "left_pos_y", "left_pos_z", ...
    "hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w"];
names = string(tableData.Properties.VariableNames);
for column = supportingColumns
    if ~ismember(column, names)
        if any(column == numericColumns)
            tableData.(char(column)) = NaN(height(tableData), 1);
        else
            tableData.(char(column)) = strings(height(tableData), 1);
        end
        warnings(end+1) = "missing_tracker_column:" + column; %#ok<AGROW>
    end
end
result = struct("table", tableData, "warnings", localUniqueWarnings(warnings), ...
    "malformed_time_header_repaired", repaired);
end

function series = localPrepareHandMotion(tracker, hand, varargin)
hand = lower(string(hand));
if ~ismember(hand, ["left", "right"])
    error("fe01:InvalidHand", "Unsupported hand: %s", hand);
end
warnings = strings(0, 1);
columns = ["time", hand + "_pos_x", hand + "_pos_y", hand + "_pos_z"];
names = string(tracker.Properties.VariableNames);
if any(~ismember(columns, names))
    series = struct("hand", hand, "samples", localEmptyHandSamples(), ...
        "warnings", "hand_position_columns_unavailable:" + hand);
    return;
end
if isempty(varargin)
    time = localNumericVector(tracker.(char(columns(1))));
else
    time = localNumericVector(varargin{1});
end
x = localNumericVector(tracker.(char(columns(2))));
y = localNumericVector(tracker.(char(columns(3))));
z = localNumericVector(tracker.(char(columns(4))));
sampleTemplate = struct("timestamp_seconds", NaN, "x_meters", NaN, ...
    "y_meters", NaN, "z_meters", NaN, "step_distance_meters", NaN, ...
    "step_speed_meters_per_second", NaN, "is_valid", false);
samples = repmat(sampleTemplate, numel(time), 1);
previousPosition = [NaN, NaN, NaN];
previousTime = NaN;
previousValid = false;
for index = 1:numel(time)
    position = [x(index), y(index), z(index)];
    valid = isfinite(time(index)) && all(isfinite(position));
    samples(index).timestamp_seconds = time(index);
    samples(index).x_meters = x(index);
    samples(index).y_meters = y(index);
    samples(index).z_meters = z(index);
    samples(index).is_valid = valid;
    if valid && previousValid
        distance = norm(position - previousPosition);
        samples(index).step_distance_meters = distance;
        delta = time(index) - previousTime;
        if isfinite(delta) && delta > 0
            samples(index).step_speed_meters_per_second = distance / delta;
        else
            warnings(end+1) = "nonpositive_time_delta:" + hand; %#ok<AGROW>
        end
    end
    if valid
        previousPosition = position;
        previousTime = time(index);
        previousValid = true;
    else
        previousPosition = [NaN, NaN, NaN];
        previousTime = NaN;
        previousValid = false;
    end
end
series = struct("hand", hand, "samples", samples, ...
    "warnings", localUniqueWarnings(warnings));
end

function samples = localEmptyHandSamples()
samples = struct("timestamp_seconds", {}, "x_meters", {}, "y_meters", {}, ...
    "z_meters", {}, "step_distance_meters", {}, ...
    "step_speed_meters_per_second", {}, "is_valid", {});
end

function tf = localBooleanVector(values)
if islogical(values)
    tf = values(:);
else
    text = lower(strtrim(string(values)));
    tf = ismember(text, ["true", "1", "yes", "y"]);
end
end

function warnings = localUniqueWarnings(warnings)
warnings = string(warnings(:));
warnings = warnings(~ismissing(warnings) & strlength(warnings) > 0);
if isempty(warnings)
    return;
end
[~, indexes] = unique(warnings, "stable");
warnings = warnings(sort(indexes));
end

function warnings = localAddWarning(warnings, warningText)
warningText = string(warningText(:));
warningText = warningText(~ismissing(warningText) & strlength(warningText) > 0);
for index = 1:numel(warningText)
    if ~any(string(warnings) == warningText(index))
        warnings(end+1, 1) = warningText(index); %#ok<AGROW>
    end
end
end

function tf = localAllMissing(values)
if isempty(values)
    tf = true;
    return;
end
if iscell(values)
    values = string(values);
end
if isstring(values) || ischar(values)
    text = string(values);
    tf = all(ismissing(text) | strlength(strtrim(text)) == 0);
elseif isnumeric(values) || islogical(values)
    numeric = double(values);
    tf = all(~isfinite(numeric(:)));
else
    text = string(values);
    tf = all(ismissing(text) | strlength(strtrim(text)) == 0);
end
end

function result = localDetectProductGrabEvents(tracker, catalog, difficulty, language)
cache = localMakeCache(tracker);
[result, ~] = localDetectProductGrabEventsCached(tracker, catalog, difficulty, language, cache);
end

function [result, cache] = localDetectProductGrabEventsCached(tracker, catalog, difficulty, language, cache)
warnings = strings(0, 1);
segments = localEmptyGrabSegments();
for hand = ["left", "right"]
    [handSegments, warnings] = localCompleteGrabSegments(tracker, hand, catalog, cache.time_seconds, warnings);
    [handSegments, warnings] = localMergeGrabSegments(handSegments, 0.10, warnings);
    for index = 1:numel(handSegments)
        if handSegments(index).release - handSegments(index).start >= 0.20 - 1e-12
            segments(end+1) = handSegments(index); %#ok<AGROW>
        end
    end
end
if ~isempty(segments)
    starts = [segments.start]';
    rightOrder = double(string({segments.hand}') == "right");
    [~, order] = sortrows([starts, rightOrder], [1 2]);
    segments = segments(order);
end
correct = localCorrectProducts(catalog, difficulty, language);
seen = strings(0, 1);
eventTemplate = struct("canonical_product_name", "", "raw_product_label", "", ...
    "hand", "", "grab_start_seconds", NaN, "grab_release_seconds", NaN, ...
    "grab_duration_seconds", NaN, "is_on_list", false, ...
    "is_first_time_on_list", false, "overlaps_other_hand_grab", false);
events = repmat(eventTemplate, 0, 1);
for index = 1:numel(segments)
    segment = segments(index);
    overlaps = false;
    for otherIndex = 1:numel(segments)
        if otherIndex ~= index && string(segments(otherIndex).hand) ~= string(segment.hand)
            overlaps = overlaps || max(segment.start, segments(otherIndex).start) < ...
                min(segment.release, segments(otherIndex).release);
        end
    end
    onList = any(correct == string(segment.canonical_product_name));
    first = onList && ~any(seen == string(segment.canonical_product_name));
    if onList
        seen(end+1) = string(segment.canonical_product_name); %#ok<AGROW>
    end
    event = eventTemplate;
    event.canonical_product_name = string(segment.canonical_product_name);
    event.raw_product_label = string(segment.raw_product_label);
    event.hand = string(segment.hand);
    event.grab_start_seconds = segment.start;
    event.grab_release_seconds = segment.release;
    event.grab_duration_seconds = segment.release - segment.start;
    event.is_on_list = onList;
    event.is_first_time_on_list = first;
    event.overlaps_other_hand_grab = overlaps;
    events(end+1) = event; %#ok<AGROW>
end
result = struct("events", events, "warnings", localUniqueWarnings(warnings));
end

function segments = localEmptyGrabSegments()
segments = struct("canonical_product_name", {}, "raw_product_label", {}, ...
    "hand", {}, "start", {}, "release", {});
end

function [segments, warnings] = localCompleteGrabSegments(tracker, hand, catalog, times, warnings)
grabColumn = "is_grabbing_" + hand;
objectColumn = "grabbed_object_" + hand;
names = string(tracker.Properties.VariableNames);
if ~ismember(grabColumn, names) || localAllMissing(tracker.(char(grabColumn)))
    warnings = localAddWarning(warnings, "grab_signal_unavailable:" + hand);
    segments = localEmptyGrabSegments();
    return;
end
if ~ismember(objectColumn, names)
    warnings = localAddWarning(warnings, "grab_object_unavailable:" + hand);
    segments = localEmptyGrabSegments();
    return;
end
grabbing = localBooleanVector(tracker.(char(grabColumn)));
labels = strtrim(string(tracker.(char(objectColumn))));
segmentTemplate = struct("canonical_product_name", "", "raw_product_label", "", ...
    "hand", hand, "start", NaN, "release", NaN);
segments = repmat(segmentTemplate, 0, 1);
activeProduct = "";
activeRaw = "";
activeStart = NaN;
for index = 1:numel(times)
    timestamp = times(index);
    if ~isfinite(timestamp)
        if strlength(activeProduct) > 0
            warnings = localAddWarning(warnings, "missing_time_during_grab:" + hand);
            activeProduct = "";
            activeRaw = "";
            activeStart = NaN;
        end
        continue;
    end
    raw = labels(index);
    if grabbing(index) && localIsProbableProduct(raw, catalog)
        product = localCanonicalizeProduct(catalog, raw);
    else
        product = "";
    end
    if strlength(activeProduct) == 0
        if strlength(product) > 0
            activeProduct = product;
            activeRaw = raw;
            activeStart = timestamp;
        end
        continue;
    end
    if product == activeProduct
        continue;
    end
    if timestamp < activeStart
        warnings = localAddWarning(warnings, "non_monotonic_grab_time:" + hand);
    else
        segment = segmentTemplate;
        segment.canonical_product_name = activeProduct;
        segment.raw_product_label = activeRaw;
        segment.start = activeStart;
        segment.release = timestamp;
        segments(end+1) = segment; %#ok<AGROW>
    end
    activeProduct = "";
    activeRaw = "";
    activeStart = NaN;
    if strlength(product) > 0
        activeProduct = product;
        activeRaw = raw;
        activeStart = timestamp;
    end
end
if strlength(activeProduct) > 0
    warnings = localAddWarning(warnings, "open_ended_grab_discarded:" + hand);
end
end

function tf = localIsProbableProduct(raw, catalog)
cleaned = localCleanProductName(raw);
placeholders = ["", "nan", "none", "noobjectgrabbed", "notassigned", ...
    "untagged", "null"];
if any(cleaned == placeholders)
    tf = false;
    return;
end
if localIsCatalogProduct(catalog, cleaned)
    tf = true;
    return;
end
nonProduct = ["tablet", "samsung_tab", "cart", "shopping_basket", "basket", ...
    "mainshelf", "main shelf", "npc", "bodum", "side l", "side r", ...
    "cube", "first shelf", "second shelf", "third shelf", "fourth shelf"];
if any(cleaned == nonProduct) || ...
        ~isempty(regexp(char(cleaned), "(^|\s)(shelf|cube|npc)(\s|$)", "once")) ...
        || contains(cleaned, "business man") || startsWith(cleaned, "swpd")
    tf = false;
    return;
end
tf = true;
end

function [merged, warnings] = localMergeGrabSegments(segments, gapLimit, warnings)
if isempty(segments)
    merged = segments;
    return;
end
merged = segments(1);
for index = 2:numel(segments)
    previous = merged(end);
    current = segments(index);
    gap = current.start - previous.release;
    if gap < 0
        warnings = localAddWarning(warnings, "non_monotonic_grab_time:" + string(current.hand));
    end
    if string(current.hand) == string(previous.hand) ...
            && string(current.canonical_product_name) == string(previous.canonical_product_name) ...
            && gap >= -1e-12 && gap <= gapLimit + 1e-12
        previous.release = current.release;
        merged(end) = previous;
    else
        merged(end+1) = current; %#ok<AGROW>
    end
end
end

function result = localDetectFocusEpisodes(tracker, catalog)
cache = localMakeCache(tracker);
[result, ~] = localDetectFocusEpisodesCached(tracker, catalog, cache);
end

function [result, cache] = localDetectFocusEpisodesCached(tracker, catalog, cache)
warnings = strings(0, 1);
times = cache.time_seconds;
names = string(tracker.Properties.VariableNames);
focusName = strings(height(tracker), 1);
focusTag = strings(height(tracker), 1);
if ismember("focus_object_name", names)
    focusName = strtrim(string(tracker.focus_object_name));
else
    warnings = localAddWarning(warnings, "focus_signal_unavailable:focus_object_name");
end
if ismember("focus_object_tag", names)
    focusTag = strtrim(string(tracker.focus_object_tag));
else
    warnings = localAddWarning(warnings, "focus_signal_unavailable:focus_object_tag");
end
blink = cell(1, 2);
blinkNames = ["is_left_eye_blinking", "is_right_eye_blinking"];
for eyeIndex = 1:2
    column = blinkNames(eyeIndex);
    if ~ismember(column, names) || localAllMissing(tracker.(char(column)))
        blink{eyeIndex} = [];
        warnings = localAddWarning(warnings, "blink_signal_unavailable:" + erase(column, "is_") );
    else
        blink{eyeIndex} = localBooleanVector(tracker.(char(column)));
    end
end
episodeTemplate = struct("canonical_focus_name", "", "raw_focus_name", "", ...
    "cleaned_focus_tag", "", "raw_focus_tag", "", "focus_start_seconds", NaN, ...
    "focus_end_seconds", NaN, "focus_duration_seconds", NaN);
episodes = repmat(episodeTemplate, 0, 1);
activeKey = "";
activeRawName = "";
activeRawTag = "";
activeStart = NaN;
for index = 1:height(tracker)
    timestamp = times(index);
    if ~isfinite(timestamp)
        if strlength(activeKey) > 0
            warnings = localAddWarning(warnings, "missing_time_during_focus");
        end
        activeKey = "";
        activeRawName = "";
        activeRawTag = "";
        activeStart = NaN;
        continue;
    end
    rawName = focusName(index);
    rawTag = focusTag(index);
    cleanName = localCleanProductName(rawName);
    cleanTag = localCleanProductName(rawTag);
    placeholder = ["noobjectgrabbed", "no object grabbed", "none", "null", ...
        "nan", "undefined", "", "notassigned", "not assigned"];
    assigned = ~any(cleanName == placeholder) && ~any(cleanTag == placeholder);
    for eyeIndex = 1:2
        if ~isempty(blink{eyeIndex}) && blink{eyeIndex}(index)
            assigned = false;
        end
    end
    if assigned
        canonicalName = localCanonicalizeProduct(catalog, cleanName);
        key = canonicalName + "|" + cleanTag;
    else
        key = "";
    end
    if strlength(activeKey) == 0
        if strlength(key) > 0
            activeKey = key;
            activeRawName = rawName;
            activeRawTag = rawTag;
            activeStart = timestamp;
        end
        continue;
    end
    if key == activeKey
        continue;
    end
    if timestamp < activeStart
        warnings = localAddWarning(warnings, "non_monotonic_focus_time");
    else
        parts = split(activeKey, "|");
        episode = episodeTemplate;
        episode.canonical_focus_name = parts(1);
        episode.raw_focus_name = activeRawName;
        episode.cleaned_focus_tag = parts(2);
        episode.raw_focus_tag = activeRawTag;
        episode.focus_start_seconds = activeStart;
        episode.focus_end_seconds = timestamp;
        episode.focus_duration_seconds = timestamp - activeStart;
        episodes(end+1) = episode; %#ok<AGROW>
    end
    activeKey = "";
    activeRawName = "";
    activeRawTag = "";
    activeStart = NaN;
    if strlength(key) > 0
        activeKey = key;
        activeRawName = rawName;
        activeRawTag = rawTag;
        activeStart = timestamp;
    end
end
if strlength(activeKey) > 0
    warnings = localAddWarning(warnings, "open_ended_focus_discarded");
end
if ~isempty(episodes)
    [~, order] = sort([episodes.focus_start_seconds]);
    episodes = episodes(order);
end
result = struct("episodes", episodes, "warnings", localUniqueWarnings(warnings));
end

function visits = localDetectListVisits(focusEpisodes)
template = struct("list_visit_start_seconds", NaN, "list_visit_end_seconds", NaN, ...
    "list_visit_duration_seconds", NaN, "is_initial_view", false);
if isempty(focusEpisodes)
    visits = repmat(template, 0, 1);
    return;
end
selected = repmat(template, 0, 1);
for index = 1:numel(focusEpisodes)
    episode = focusEpisodes(index);
    if string(episode.cleaned_focus_tag) == "tablet" ...
            || any(string(episode.canonical_focus_name) == ["tablet", "samsung_tab"])
        visit = template;
        visit.list_visit_start_seconds = episode.focus_start_seconds;
        visit.list_visit_end_seconds = episode.focus_end_seconds;
        visit.list_visit_duration_seconds = episode.focus_duration_seconds;
        selected(end+1) = visit; %#ok<AGROW>
    end
end
if ~isempty(selected)
    [~, order] = sort([selected.list_visit_start_seconds]);
    selected = selected(order);
    for index = 1:numel(selected)
        selected(index).is_initial_view = index == 1;
    end
end
visits = selected;
end

function result = localDetectSearchIntervals(tracker, grabEvents, listVisits, focusEpisodes)
cache = localMakeCache(tracker);
[result, ~] = localDetectSearchIntervalsCached(tracker, grabEvents, listVisits, focusEpisodes, cache);
end

function [result, cache] = localDetectSearchIntervalsCached(tracker, grabEvents, listVisits, focusEpisodes, cache)
warnings = strings(0, 1);
times = cache.time_seconds;
blackout = cache.blackout_values;
if isempty(blackout)
    warnings = localAddWarning(warnings, "usable_activity_start_unavailable");
end
usableStart = NaN;
if ~isempty(blackout)
    valid = isfinite(times) & ~blackout;
    first = find(valid, 1, "first");
    if ~isempty(first)
        usableStart = times(first);
    else
        warnings = localAddWarning(warnings, "usable_activity_start_unavailable");
    end
end
if isempty(blackout) || ~isfinite(usableStart)
    usableStart = NaN;
end
qualifying = grabEvents([grabEvents.is_on_list] & [grabEvents.is_first_time_on_list]);
if ~isempty(qualifying)
    [~, order] = sort([qualifying.grab_start_seconds]);
    qualifying = qualifying(order);
end
visits = listVisits;
if ~isempty(visits)
    [~, order] = sort([visits.list_visit_start_seconds]);
    visits = visits(order);
end
[focusEpisodes, cache] = localSortedFocus(focusEpisodes, cache);
intervalTemplate = struct("canonical_product_name", "", ...
    "qualifying_grab_start_seconds", NaN, "search_start_seconds", NaN, ...
    "first_target_focus_seconds", NaN, "is_valid", false);
intervals = repmat(intervalTemplate, 0, 1);
previousGrab = NaN;
for index = 1:numel(qualifying)
    event = qualifying(index);
    grabStart = event.grab_start_seconds;
    completedEnds = [visits.list_visit_end_seconds];
    completedEnds = completedEnds(isfinite(completedEnds) & completedEnds <= grabStart);
    latestVisitEnd = max(completedEnds, [], "omitnan");
    if isempty(completedEnds)
        latestVisitEnd = NaN;
    end
    if ~isfinite(previousGrab)
        if ~isfinite(usableStart)
            interval = intervalTemplate;
            interval.canonical_product_name = event.canonical_product_name;
            interval.qualifying_grab_start_seconds = grabStart;
            intervals(end+1) = interval; %#ok<AGROW>
            previousGrab = grabStart;
            continue;
        end
        if isfinite(latestVisitEnd)
            searchStart = max(usableStart, latestVisitEnd);
        else
            searchStart = usableStart;
        end
    else
        if isfinite(latestVisitEnd)
            searchStart = max(previousGrab, latestVisitEnd);
        else
            searchStart = previousGrab;
        end
    end
    if searchStart > grabStart
        warnings = localAddWarning(warnings, "search_start_after_grab");
        interval = intervalTemplate;
        interval.canonical_product_name = event.canonical_product_name;
        interval.qualifying_grab_start_seconds = grabStart;
        intervals(end+1) = interval; %#ok<AGROW>
        previousGrab = grabStart;
        continue;
    end
    candidates = NaN(0, 1);
    for focusIndex = 1:numel(focusEpisodes)
        episode = focusEpisodes(focusIndex);
        if episode.focus_start_seconds > grabStart
            break;
        end
        if string(episode.canonical_focus_name) == string(event.canonical_product_name) ...
                && episode.focus_end_seconds >= searchStart
            candidate = max(searchStart, episode.focus_start_seconds);
            if candidate <= grabStart
                candidates(end+1) = candidate; %#ok<AGROW>
            end
        end
    end
    interval = intervalTemplate;
    interval.canonical_product_name = event.canonical_product_name;
    interval.qualifying_grab_start_seconds = grabStart;
    interval.search_start_seconds = searchStart;
    if ~isempty(candidates)
        interval.first_target_focus_seconds = min(candidates);
        interval.is_valid = true;
    else
        warnings = localAddWarning(warnings, "target_focus_not_found");
    end
    intervals(end+1) = interval; %#ok<AGROW>
    previousGrab = grabStart;
end
result = struct("intervals", intervals, "warnings", localUniqueWarnings(warnings));
end

function result = localDetectReachIntervals(tracker, grabEvents, varargin)
searchIntervals = repmat(struct("canonical_product_name", "", ...
    "qualifying_grab_start_seconds", NaN, "search_start_seconds", NaN, ...
    "first_target_focus_seconds", NaN, "is_valid", false), 0, 1);
absoluteThreshold = 0.05;
peakFraction = 0.05;
minimumDuration = 0.10;
for index = 1:2:numel(varargin)
    name = lower(string(varargin{index}));
    value = varargin{index+1};
    switch name
        case "searchintervals"
            searchIntervals = value;
        case "absolutespeedthresholdmeterspersecond"
            absoluteThreshold = double(value);
        case "peakspeedfraction"
            peakFraction = double(value);
        case "minimummovementdurationseconds"
            minimumDuration = double(value);
        otherwise
            error("fe01:UnknownOption", "Unknown reach option: %s", name);
    end
end
cache = localMakeCache(tracker);
[result, ~] = localDetectReachIntervalsCached(tracker, grabEvents, searchIntervals, ...
    absoluteThreshold, peakFraction, minimumDuration, cache);
end

function [result, cache] = localDetectReachIntervalsCached(tracker, grabEvents, searchIntervals, absoluteThreshold, peakFraction, minimumDuration, cache)
if absoluteThreshold < 0 || peakFraction < 0 || peakFraction > 1 || minimumDuration < 0
    error("fe01:InvalidReachConfig", "Invalid movement threshold configuration.");
end
warnings = strings(0, 1);
[searchMap, searchWarnings] = localIntervalMap(searchIntervals, ...
    "qualifying_grab_start_seconds", "reach_detection");
warnings = [warnings; string(searchWarnings(:))]; %#ok<AGROW>
qualifying = grabEvents([grabEvents.is_on_list] & [grabEvents.is_first_time_on_list]);
if ~isempty(qualifying)
    [~, order] = sort([qualifying.grab_start_seconds]);
    qualifying = qualifying(order);
end
intervalTemplate = struct("canonical_product_name", "", "hand", "", ...
    "reach_start_seconds", NaN, "grab_start_seconds", NaN, "is_valid", false);
intervals = repmat(intervalTemplate, 0, 1);
for eventIndex = 1:numel(qualifying)
    event = qualifying(eventIndex);
    hand = string(event.hand);
    grabStart = event.grab_start_seconds;
    key = localEventKey(event.canonical_product_name, grabStart);
    if isempty(key) || ~isKey(searchMap, key)
        warnings = localAddWarning(warnings, "reach_search_interval_unavailable:" + hand);
        interval = intervalTemplate;
        interval.canonical_product_name = event.canonical_product_name;
        interval.hand = hand;
        interval.grab_start_seconds = grabStart;
        intervals(end+1) = interval; %#ok<AGROW>
        continue;
    end
    search = searchMap(key);
    searchStart = search.search_start_seconds;
    if ~isfield(search, "is_valid") || ~search.is_valid
        warnings = localAddWarning(warnings, "reach_search_interval_unavailable:" + hand);
        interval = intervalTemplate;
        interval.canonical_product_name = event.canonical_product_name;
        interval.hand = hand;
        interval.grab_start_seconds = grabStart;
        intervals(end+1) = interval; %#ok<AGROW>
        continue;
    end
    if ~isfinite(searchStart) || ~isfinite(grabStart) || searchStart > grabStart
        warnings = localAddWarning(warnings, "reach_interval_boundaries_invalid:" + hand);
        interval = intervalTemplate;
        interval.canonical_product_name = event.canonical_product_name;
        interval.hand = hand;
        interval.grab_start_seconds = grabStart;
        intervals(end+1) = interval; %#ok<AGROW>
        continue;
    end
    [motion, cache] = localCachedHandMotion(tracker, hand, cache);
    warnings = [warnings; string(motion.warnings(:))]; %#ok<AGROW>
    candidates = motion.samples(isfinite([motion.samples.timestamp_seconds]) ...
        & [motion.samples.timestamp_seconds] >= searchStart ...
        & [motion.samples.timestamp_seconds] <= grabStart);
    speeds = [candidates.step_speed_meters_per_second];
    speeds = speeds(isfinite(speeds));
    interval = intervalTemplate;
    interval.canonical_product_name = event.canonical_product_name;
    interval.hand = hand;
    interval.grab_start_seconds = grabStart;
    if isempty(speeds)
        warnings = localAddWarning(warnings, "reach_onset_not_found:" + hand);
        intervals(end+1) = interval; %#ok<AGROW>
        continue;
    end
    threshold = max(absoluteThreshold, peakFraction * max(speeds));
    sampleIndex = find(isfinite([motion.samples.timestamp_seconds]) ...
        & [motion.samples.timestamp_seconds] >= searchStart ...
        & [motion.samples.timestamp_seconds] <= grabStart);
    movementBlocks = {};
    blockOnsets = NaN(0, 1);
    active = [];
    activeOnset = NaN;
    previousTimestamp = NaN;
    for candidateIndex = 1:numel(sampleIndex)
        fullIndex = sampleIndex(candidateIndex);
        sample = motion.samples(fullIndex);
        timestamp = sample.timestamp_seconds;
        speed = sample.step_speed_meters_per_second;
        if ~sample.is_valid || ~isfinite(timestamp) || ~isfinite(speed)
            if ~isempty(active)
                movementBlocks{end+1} = active; %#ok<AGROW>
                blockOnsets(end+1) = activeOnset; %#ok<AGROW>
            end
            active = [];
            activeOnset = NaN;
            previousTimestamp = NaN;
            continue;
        end
        if isfinite(previousTimestamp) && timestamp <= previousTimestamp
            warnings = localAddWarning(warnings, "non_monotonic_reach_time:" + hand);
            if ~isempty(active)
                movementBlocks{end+1} = active; %#ok<AGROW>
                blockOnsets(end+1) = activeOnset; %#ok<AGROW>
            end
            active = [];
            activeOnset = NaN;
        end
        if speed >= threshold
            if isempty(active)
                predecessor = fullIndex - 1;
                if predecessor < 1 || ~motion.samples(predecessor).is_valid ...
                        || ~isfinite(motion.samples(predecessor).timestamp_seconds)
                    warnings = localAddWarning(warnings, "reach_onset_predecessor_unavailable:" + hand);
                    activeOnset = NaN;
                else
                    activeOnset = max(searchStart, motion.samples(predecessor).timestamp_seconds);
                end
            end
            active(end+1) = fullIndex; %#ok<AGROW>
        else
            if ~isempty(active)
                movementBlocks{end+1} = active; %#ok<AGROW>
                blockOnsets(end+1) = activeOnset; %#ok<AGROW>
            end
            active = [];
            activeOnset = NaN;
        end
        previousTimestamp = timestamp;
    end
    if ~isempty(active)
        movementBlocks{end+1} = active; %#ok<AGROW>
        blockOnsets(end+1) = activeOnset; %#ok<AGROW>
    end
    qualifyingBlocks = false(1, numel(movementBlocks));
    for blockIndex = 1:numel(movementBlocks)
        block = movementBlocks{blockIndex};
        if ~isfinite(blockOnsets(blockIndex)) || isempty(block)
            continue;
        end
        blockDuration = motion.samples(block(end)).timestamp_seconds - ...
            motion.samples(block(1)).timestamp_seconds;
        qualifyingBlocks(blockIndex) = blockDuration + 1e-12 >= minimumDuration;
    end
    selected = find(qualifyingBlocks, 1, "last");
    if isempty(selected)
        warnings = localAddWarning(warnings, "reach_onset_not_found:" + hand);
    else
        interval.reach_start_seconds = blockOnsets(selected);
        interval.is_valid = true;
    end
    intervals(end+1) = interval; %#ok<AGROW>
end
result = struct("intervals", intervals, "warnings", localUniqueWarnings(warnings));
end

function result = localAggregatePace(grabEvents, listVisits)
qualifying = grabEvents([grabEvents.is_on_list] & [grabEvents.is_first_time_on_list]);
starts = sort([qualifying.grab_start_seconds]);
if numel(starts) < 2
    pace = NaN;
else
    pace = median(diff(starts));
end
rechecks = listVisits(~[listVisits.is_initial_view]);
result = struct("median_time_between_qualifying_grabs_seconds", pace, ...
    "list_recheck_count", numel(rechecks), ...
    "total_list_recheck_duration_seconds", sum([rechecks.list_visit_duration_seconds], "omitnan"));
end

function cache = localMakeCache(tracker)
cache = struct();
cache.time_seconds = localNumericVector(tracker.time);
cache.blackout_values = [];
if ismember("enableBlackout", string(tracker.Properties.VariableNames)) && ...
        ~localAllMissing(tracker.enableBlackout)
    cache.blackout_values = localBooleanVector(tracker.enableBlackout);
end
cache.hand_motion = {[], []};
cache.hand_prepared = [false, false];
cache.head_rotations = [];
cache.head_rotation_prepared = false;
cache.head_rotation_missing = false;
cache.sorted_focus = [];
cache.focus_source_count = -1;
end

function [motion, cache] = localCachedHandMotion(tracker, hand, cache)
index = 1 + double(string(hand) == "right");
if ~cache.hand_prepared(index)
    motion = localPrepareHandMotion(tracker, hand, cache.time_seconds);
    cache.hand_motion{index} = motion;
    cache.hand_prepared(index) = true;
else
    motion = cache.hand_motion{index};
end
end

function [focusEpisodes, cache] = localSortedFocus(focusEpisodes, cache)
if cache.focus_source_count ~= numel(focusEpisodes)
    if isempty(focusEpisodes)
        cache.sorted_focus = focusEpisodes;
    else
        [~, order] = sort([focusEpisodes.focus_start_seconds]);
        cache.sorted_focus = focusEpisodes(order);
    end
    cache.focus_source_count = numel(focusEpisodes);
end
focusEpisodes = cache.sorted_focus;
end

function [map, warnings] = localIntervalMap(intervals, timeField, varargin)
map = containers.Map("KeyType", "char", "ValueType", "any");
warnings = strings(0, 1);
mode = "builder";
if ~isempty(varargin)
    mode = lower(string(varargin{1}));
end
if mode == "reach_detection"
    keyWarning = "reach_search_interval_key_unavailable";
    duplicatePrefix = "duplicate_reach_search_interval";
elseif timeField == "qualifying_grab_start_seconds"
    keyWarning = "search_interval_key_unavailable";
    duplicatePrefix = "duplicate_search_interval";
else
    keyWarning = "reach_interval_key_unavailable";
    duplicatePrefix = "duplicate_reach_interval";
end
for index = 1:numel(intervals)
    interval = intervals(index);
    if ~isfield(interval, timeField) || ~isfield(interval, "canonical_product_name")
        warnings = localAddWarning(warnings, keyWarning);
        continue;
    end
    key = localEventKey(interval.canonical_product_name, interval.(timeField));
    if isempty(key)
        warnings = localAddWarning(warnings, keyWarning);
    elseif isKey(map, key)
        warnings = localAddWarning(warnings, duplicatePrefix + ":" + ...
            localCleanProductName(interval.canonical_product_name) + ":" + ...
            string(localNumericScalar(interval.(timeField))));
    else
        map(key) = interval;
    end
end
warnings = localUniqueWarnings(warnings);
end

function key = localEventKey(product, timestamp)
value = localNumericScalar(timestamp);
if ~isfinite(value)
    key = "";
else
    key = sprintf("%s|%.17g", char(localCleanProductName(product)), value);
end
end

function result = localAggregateLocating(searchIntervals, focusEpisodes, tracker, catalog)
cache = localMakeCache(tracker);
[result, ~] = localAggregateLocatingCached(searchIntervals, focusEpisodes, tracker, catalog, cache);
end

function [result, cache] = localAggregateLocatingCached(searchIntervals, focusEpisodes, tracker, catalog, cache)
warnings = strings(0, 1);
if isempty(searchIntervals)
    result = struct("median_time_to_target_seconds", NaN, ...
        "median_irrelevant_focus_duration_seconds", NaN, ...
        "median_head_turning_degrees", NaN, "warnings", strings(0, 1));
    return;
end
[focusEpisodes, cache] = localSortedFocus(focusEpisodes, cache);
locating = NaN(0, 1);
irrelevant = NaN(0, 1);
turning = NaN(0, 1);
for intervalIndex = 1:numel(searchIntervals)
    interval = searchIntervals(intervalIndex);
    startTime = interval.search_start_seconds;
    targetTime = interval.first_target_focus_seconds;
    if ~interval.is_valid || ~isfinite(startTime) || ~isfinite(targetTime)
        warnings = localAddWarning(warnings, "locating_time_unavailable");
        continue;
    end
    duration = targetTime - startTime;
    if duration < 0 || ~isfinite(duration)
        warnings = localAddWarning(warnings, "negative_locating_duration");
    else
        locating(end+1) = duration; %#ok<AGROW>
    end
    targetName = localCleanProductName(interval.canonical_product_name);
    irrelevantDuration = 0;
    labelsSeen = strings(0, 1);
    for focusIndex = 1:numel(focusEpisodes)
        episode = focusEpisodes(focusIndex);
        if episode.focus_start_seconds > targetTime
            break;
        end
        overlapStart = max(startTime, episode.focus_start_seconds);
        overlapEnd = min(targetTime, episode.focus_end_seconds);
        if overlapEnd <= overlapStart
            continue;
        end
        if localExplicitIrrelevant(episode)
            irrelevantDuration = irrelevantDuration + (overlapEnd - overlapStart);
            continue;
        end
        if localExcludedFocus(episode, targetName, catalog)
            continue;
        end
        label = localCleanProductName(episode.canonical_focus_name);
        if strlength(label) == 0
            label = localCleanProductName(episode.cleaned_focus_tag);
        end
        if ~any(labelsSeen == label)
            labelsSeen(end+1) = label; %#ok<AGROW>
            warnings = localAddWarning(warnings, "unclassifiable_focus:" + label);
        end
        continue;
    end
    irrelevant(end+1) = irrelevantDuration; %#ok<AGROW>
    [pathDegrees, cache, rotationWarnings] = localHeadRotationPathCached(tracker, startTime, targetTime, cache);
    warnings = [warnings; rotationWarnings(:)]; %#ok<AGROW>
    if isfinite(pathDegrees)
        turning(end+1) = pathDegrees; %#ok<AGROW>
    end
end
result = struct("median_time_to_target_seconds", localMedianOrNaN(locating), ...
    "median_irrelevant_focus_duration_seconds", localMedianOrNaN(irrelevant), ...
    "median_head_turning_degrees", localMedianOrNaN(turning), ...
    "warnings", localUniqueWarnings(warnings));
end

function tf = localExcludedFocus(episode, targetName, catalog)
name = localCleanProductName(episode.canonical_focus_name);
tag = localCleanProductName(episode.cleaned_focus_tag);
if strlength(name) == 0 && strlength(tag) == 0
    tf = true;
    return;
end
if name == targetName || tag == targetName
    tf = true;
    return;
end
if localIsCatalogProduct(catalog, name) || localIsCatalogProduct(catalog, tag)
    tf = true;
    return;
end
excluded = ["", "nan", "none", "noobjectgrabbed", "notassigned", ...
    "untagged", "null", "tablet", "samsung_tab", "cart", ...
    "shopping_basket", "basket", "mainshelf", "main shelf", "npc", ...
    "bodum", "side l", "side r", "cube", "first shelf", ...
    "second shelf", "third shelf", "fourth shelf"];
tf = any(name == excluded) || any(tag == excluded);
end

function tf = localExplicitIrrelevant(episode)
name = localCleanProductName(episode.canonical_focus_name);
tag = localCleanProductName(episode.cleaned_focus_tag);
explicit = ["npc", "bodum", "side l", "side r"];
tf = any(name == explicit) || any(tag == explicit);
end

function [degrees, cache, warnings] = localHeadRotationPathCached(tracker, startTime, endTime, cache)
warnings = strings(0, 1);
if ~cache.head_rotation_prepared
    required = ["hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w"];
    names = string(tracker.Properties.VariableNames);
    if any(~ismember(required, names))
        cache.head_rotation_missing = true;
        cache.head_rotations = zeros(0, 5);
    else
        time = cache.time_seconds;
        components = zeros(numel(time), 4);
        for componentIndex = 1:4
            components(:, componentIndex) = localNumericVector(tracker.(char(required(componentIndex))));
        end
        valid = isfinite(time) & all(isfinite(components), 2);
        norms = sqrt(sum(components.^2, 2));
        valid = valid & isfinite(norms) & norms > 0;
        rotations = components(valid, :) ./ norms(valid);
        cache.head_rotations = [time(valid), rotations];
    end
    cache.head_rotation_prepared = true;
end
if cache.head_rotation_missing
    warnings(end+1) = "head_rotation_unavailable";
    degrees = NaN;
    return;
end
window = cache.head_rotations(cache.head_rotations(:,1) >= startTime & ...
    cache.head_rotations(:,1) <= endTime, :);
if size(window, 1) < 2
    warnings(end+1) = "head_rotation_unavailable";
    degrees = NaN;
    return;
end
degrees = 0;
for index = 2:size(window, 1)
    dotValue = abs(dot(window(index-1, 2:5), window(index, 2:5)));
    dotValue = min(1, max(-1, dotValue));
    degrees = degrees + 2 * rad2deg(acos(dotValue));
end
if ~isfinite(degrees)
    warnings(end+1) = "head_rotation_unavailable";
    degrees = NaN;
end
end

function result = localPrepareHeadsetRotations(tracker)
%LOCALPREPAREHEADSETROTATIONS Normalize valid recorded headset quaternions.
% The returned rotations matrix is [time qx qy qz qw]. Invalid rows are
% omitted from the normalized matrix and sample list, matching the Python
% within-trial cache representation without inventing timestamps.
warnings = strings(0, 1);
required = ["time", "hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w"];
names = string(tracker.Properties.VariableNames);
sampleTemplate = struct("timestamp_seconds", NaN, ...
    "qx", NaN, "qy", NaN, "qz", NaN, "qw", NaN, ...
    "quaternion_x", NaN, "quaternion_y", NaN, ...
    "quaternion_z", NaN, "quaternion_w", NaN, "is_valid", false);
samples = repmat(sampleTemplate, 0, 1);
rotations = zeros(0, 5);
if any(~ismember(required, names))
    warnings = localAddWarning(warnings, "head_rotation_unavailable");
    result = struct("samples", samples, "rotations", rotations, ...
        "warnings", localUniqueWarnings(warnings));
    return;
end

time = localNumericVector(tracker.time);
components = zeros(numel(time), 4);
for componentIndex = 1:4
    components(:, componentIndex) = localNumericVector( ...
        tracker.(char(required(componentIndex + 1))));
end
for index = 1:numel(time)
    quaternion = components(index, :);
    normValue = sqrt(sum(quaternion.^2));
    if ~isfinite(time(index)) || any(~isfinite(quaternion)) || ...
            ~isfinite(normValue) || normValue <= 0
        continue;
    end
    quaternion = quaternion ./ normValue;
    sample = sampleTemplate;
    sample.timestamp_seconds = time(index);
    sample.qx = quaternion(1);
    sample.qy = quaternion(2);
    sample.qz = quaternion(3);
    sample.qw = quaternion(4);
    sample.quaternion_x = quaternion(1);
    sample.quaternion_y = quaternion(2);
    sample.quaternion_z = quaternion(3);
    sample.quaternion_w = quaternion(4);
    sample.is_valid = true;
    samples(end+1, 1) = sample; %#ok<AGROW>
end
if ~isempty(samples)
    rotations = [[samples.timestamp_seconds]', [samples.qx]', [samples.qy]', ...
        [samples.qz]', [samples.qw]'];
end
result = struct("samples", samples, "rotations", rotations, ...
    "warnings", localUniqueWarnings(warnings));
end

function result = localAggregateReach(reachIntervals, tracker, varargin)
minimumStraight = 0.02;
for index = 1:2:numel(varargin)
    option = lower(string(varargin{index}));
    if option == "minimumstraightdistancemeters"
        minimumStraight = double(varargin{index+1});
    else
        error("fe01:UnknownOption", "Unknown reach aggregation option: %s", option);
    end
end
cache = localMakeCache(tracker);
[result, ~] = localAggregateReachCached(reachIntervals, tracker, minimumStraight, cache);
end

function [result, cache] = localAggregateReachCached(reachIntervals, tracker, minimumStraight, cache)
if minimumStraight < 0
    error("fe01:InvalidReachConfig", "Minimum straight distance cannot be negative.");
end
warnings = strings(0, 1);
durations = NaN(0, 1);
ratios = NaN(0, 1);
for index = 1:numel(reachIntervals)
    interval = reachIntervals(index);
    grabStart = interval.grab_start_seconds;
    reachStart = interval.reach_start_seconds;
    if ~interval.is_valid || ~isfinite(reachStart) || ~isfinite(grabStart)
        warnings = localAddWarning(warnings, "reach_duration_unavailable");
        continue;
    end
    duration = grabStart - reachStart;
    if ~isfinite(duration) || duration < 0
        warnings = localAddWarning(warnings, "negative_reach_duration");
    else
        durations(end+1) = duration; %#ok<AGROW>
    end
    [motion, cache] = localCachedHandMotion(tracker, interval.hand, cache);
    warnings = [warnings; string(motion.warnings(:))]; %#ok<AGROW>
    [ratio, warnings] = localReachPathRatio( ...
        motion, reachStart, grabStart, minimumStraight, interval.hand, warnings);
    if isfinite(ratio)
        ratios(end+1) = ratio; %#ok<AGROW>
    end
end
result = struct("median_reach_duration_seconds", localMedianOrNaN(durations), ...
    "median_reach_path_ratio", localMedianOrNaN(ratios), ...
    "warnings", localUniqueWarnings(warnings));
end

function [ratio, warnings] = localReachPathRatio(motion, reachStart, grabStart, minimumStraight, hand, warnings)
ratio = NaN;
samples = motion.samples;
if isempty(samples)
    warnings = localAddWarning(warnings, "insufficient_reach_path:" + hand);
    return;
end
timestamps = [samples.timestamp_seconds];
windowIndexes = find(isfinite(timestamps) & timestamps >= reachStart & timestamps <= grabStart);
validIndexes = windowIndexes([samples(windowIndexes).is_valid]);
if numel(validIndexes) < 2
    warnings = localAddWarning(warnings, "insufficient_reach_positions:" + hand);
    return;
end
if any(~[samples(windowIndexes).is_valid])
    warnings = localAddWarning(warnings, "reach_position_gap:" + hand);
end
first = samples(validIndexes(1));
last = samples(validIndexes(end));
firstPosition = [first.x_meters, first.y_meters, first.z_meters];
lastPosition = [last.x_meters, last.y_meters, last.z_meters];
if ~all(isfinite(firstPosition)) || ~all(isfinite(lastPosition))
    warnings = localAddWarning(warnings, "reach_endpoint_position_missing:" + hand);
    return;
end
straight = norm(lastPosition - firstPosition);
if ~isfinite(straight)
    warnings = localAddWarning(warnings, "reach_endpoint_position_missing:" + hand);
    return;
end
if straight < minimumStraight
    warnings = localAddWarning(warnings, "straight_distance_too_small");
    return;
end
pathSteps = NaN(0, 1);
for index = 2:numel(windowIndexes)
    previous = samples(windowIndexes(index-1));
    current = samples(windowIndexes(index));
    if previous.is_valid && current.is_valid && isfinite(previous.timestamp_seconds) ...
            && isfinite(current.timestamp_seconds) && isfinite(current.step_distance_meters)
        pathSteps(end+1) = current.step_distance_meters; %#ok<AGROW>
    end
end
if isempty(pathSteps)
    warnings = localAddWarning(warnings, "insufficient_reach_path:" + hand);
    return;
end
pathLength = sum(pathSteps);
ratio = pathLength / straight;
if ~isfinite(ratio)
    warnings = localAddWarning(warnings, "reach_path_ratio_unavailable:" + hand);
    ratio = NaN;
else
    ratio = max(1, ratio);
end
end

function result = localDeriveErrorOutcomes(performance)
errorAliases = containers.Map( ...
    {"errors_total", "total_error_count", "errors_missing", "errors_wrongorder", ...
    "errors_wrong_order", "errors_collectedmorethanonce", "errors_duplicate", ...
    "errors_notinlist", "errors_not_in_list"}, ...
    {"errors_total", "errors_total", "errors_missing", "errors_wrong_order", ...
    "errors_wrong_order", "errors_duplicate", "errors_duplicate", ...
    "errors_not_in_list", "errors_not_in_list"});
normalized = performance;
names = string(normalized.Properties.VariableNames);
for index = 1:numel(names)
    key = lower(strtrim(names(index)));
    if isKey(errorAliases, char(key))
        normalized.Properties.VariableNames{index} = errorAliases(char(key));
    end
end
normalized = localStandardizeTrialTable(normalized);
errorColumns = ["errors_missing", "errors_wrong_order", "errors_duplicate", "errors_not_in_list"];
for column = [errorColumns, "errors_total"]
    if ~ismember(column, string(normalized.Properties.VariableNames))
        normalized.(char(column)) = NaN(height(normalized), 1);
    else
        normalized.(char(column)) = localNumericVector(normalized.(char(column)));
    end
end
recordTemplate = struct("participant_id", "", "session_id", "", ...
    "condition_name", "", "difficulty_level", NaN, "trial_order", "", ...
    "errors_missing", NaN, "errors_wrong_order", NaN, "errors_duplicate", NaN, ...
    "errors_not_in_list", NaN, "total_error_count", NaN, ...
    "error_change_from_d0", NaN);
records = repmat(recordTemplate, 0, 1);
warnings = strings(0, 1);
for rowIndex = 1:height(normalized)
    components = zeros(1, 4);
    for componentIndex = 1:4
        components(componentIndex) = normalized{rowIndex, char(errorColumns(componentIndex))};
    end
    componentSum = NaN;
    if all(isfinite(components))
        componentSum = sum(components);
    end
    recordedTotal = normalized{rowIndex, "errors_total"};
    if ~isfinite(recordedTotal)
        total = componentSum;
    else
        total = recordedTotal;
        if isfinite(componentSum) && ...
                abs(total - componentSum) > 1e-8 + 1e-5 * abs(componentSum)
            warnings = localAddWarning(warnings, "error_total_component_mismatch");
        end
    end
    if ~isfinite(total)
        warnings = localAddWarning(warnings, "error_total_unavailable");
    end
    record = recordTemplate;
    record.participant_id = string(normalized.participant_id(rowIndex));
    record.session_id = string(normalized.session_id(rowIndex));
    record.condition_name = string(normalized.condition_name(rowIndex));
    record.difficulty_level = normalized.difficulty_level(rowIndex);
    record.trial_order = string(normalized.trial_order(rowIndex));
    record.errors_missing = normalized{rowIndex, "errors_missing"};
    record.errors_wrong_order = normalized{rowIndex, "errors_wrong_order"};
    record.errors_duplicate = normalized{rowIndex, "errors_duplicate"};
    record.errors_not_in_list = normalized{rowIndex, "errors_not_in_list"};
    record.total_error_count = total;
    records(end+1) = record; %#ok<AGROW>
end
keys = strings(0, 1);
values = cell(0, 1);
for index = 1:numel(records)
    record = records(index);
    if record.difficulty_level == 0 && isfinite(record.total_error_count)
        key = localParticipantConditionKey(record.participant_id, record.condition_name);
        position = find(keys == key, 1, "first");
        if isempty(position)
            keys(end+1) = key; %#ok<AGROW>
            values{end+1} = record.total_error_count; %#ok<AGROW>
        else
            values{position}(end+1) = record.total_error_count;
        end
    end
end
baselines = containers.Map("KeyType", "char", "ValueType", "double");
for index = 1:numel(keys)
    baselineValues = values{index};
    if numel(baselineValues) > 1
        parts = split(keys(index), "|");
        warnings = localAddWarning(warnings, "duplicate_d0_baseline:" + parts(1) + ":" + parts(2));
    end
    baselines(char(keys(index))) = median(baselineValues);
end
missingBaselineKeys = strings(0, 1);
for index = 1:numel(records)
    difficulty = records(index).difficulty_level;
    if difficulty == 0
        continue;
    end
    key = localParticipantConditionKey(records(index).participant_id, records(index).condition_name);
    if ~ismember(difficulty, [2, 6, 10])
        warnings = localAddWarning(warnings, "unsupported_difficulty:" + string(difficulty));
        continue;
    end
    if ~isKey(baselines, char(key)) || ~isfinite(records(index).total_error_count)
        if ~isKey(baselines, char(key)) && ~any(missingBaselineKeys == key)
            parts = split(key, "|");
            warnings = localAddWarning(warnings, "d0_baseline_unavailable:" + parts(1) + ":" + parts(2));
            missingBaselineKeys(end+1) = key; %#ok<AGROW>
        end
        continue;
    end
    records(index).error_change_from_d0 = records(index).total_error_count - baselines(char(key));
end
result = struct("records", records, "warnings", localUniqueWarnings(warnings));
end

function [tableData, warnings, units] = localDerivePerformanceMetrics(trials)
output = trials;
warnings = strings(0, 1);
output.correct_products_collected_count = NaN(height(output), 1);
output.performance_percent = NaN(height(output), 1);
output.performance_change_from_d0_percentage_points = NaN(height(output), 1);
names = lower(strtrim(string(output.Properties.VariableNames)));
explicitNames = ["performance_percent", "performance_percentage", "performance_pct", "percentage_performance"];
explicitIndex = find(ismember(names, explicitNames), 1, "first");
if ~isempty(explicitIndex)
    percent = localNumericVector(output{:, explicitIndex});
    invalid = ~isfinite(percent) & ~localAllMissing(output{:, explicitIndex});
    if any(invalid)
        warnings = localAddWarning(warnings, "invalid_performance_percent_values");
    end
    outOfRange = isfinite(percent) & (percent < 0 | percent > 100);
    if any(outOfRange)
        warnings = localAddWarning(warnings, "performance_percent_out_of_range");
    end
    percent(outOfRange) = NaN;
    output.performance_percent = percent;
    if ismember("performance", string(output.Properties.VariableNames))
        counts = localNumericVector(output.performance);
        counts(~isfinite(counts) | counts < 0 | counts > 20) = NaN;
        output.correct_products_collected_count = counts;
    end
    units = "explicit_percentage";
elseif ismember("performance", string(output.Properties.VariableNames))
    counts = localNumericVector(output.performance);
    finite = isfinite(counts);
    valid = finite & counts >= 0 & counts <= 20;
    if any(finite & ~valid)
        warnings = localAddWarning(warnings, "performance_units_unresolved");
    end
    normalizedCounts = counts;
    normalizedCounts(~valid) = NaN;
    output.correct_products_collected_count = normalizedCounts;
    percent = counts * 5;
    percent(~valid) = NaN;
    output.performance_percent = percent;
    warnings = localAddWarning(warnings, "performance_interpreted_as_correct_product_count");
    units = "correct_product_count";
else
    warnings = localAddWarning(warnings, "performance_source_unavailable");
    units = "unknown";
end
keys = strings(0, 1);
values = cell(0, 1);
for index = 1:height(output)
    if output.difficulty_level(index) == 0 && isfinite(output.performance_percent(index))
        key = localParticipantConditionKey(output.participant_id(index), output.condition_name(index));
        position = find(keys == key, 1, "first");
        if isempty(position)
            keys(end+1) = key; %#ok<AGROW>
            values{end+1} = output.performance_percent(index); %#ok<AGROW>
        else
            values{position}(end+1) = output.performance_percent(index);
        end
    end
end
baselines = containers.Map("KeyType", "char", "ValueType", "double");
for index = 1:numel(keys)
    if numel(values{index}) > 1
        parts = split(keys(index), "|");
        warnings = localAddWarning(warnings, "duplicate_performance_d0_baseline:" + parts(1) + ":" + parts(2));
    end
    baselines(char(keys(index))) = median(values{index});
end
for index = 1:height(output)
    difficulty = output.difficulty_level(index);
    if difficulty == 0
        continue;
    end
    if ~ismember(difficulty, [2, 6, 10])
        warnings = localAddWarning(warnings, "unsupported_performance_difficulty:" + string(difficulty));
        continue;
    end
    key = localParticipantConditionKey(output.participant_id(index), output.condition_name(index));
    if isKey(baselines, char(key)) && isfinite(output.performance_percent(index))
        output.performance_change_from_d0_percentage_points(index) = ...
            baselines(char(key)) - output.performance_percent(index);
    end
end
end

function key = localParticipantConditionKey(participant, condition)
key = string(participant) + "|" + string(condition);
end

function value = localMedianOrNaN(values)
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = median(values);
end
end

function result = localBuildProductGrabFeatures(trialMetadata, grabEvents, searchIntervals, reachIntervals, tracker)
cache = localMakeCache(tracker);
[result, ~] = localBuildProductGrabFeaturesCached(trialMetadata, grabEvents, searchIntervals, reachIntervals, tracker, cache);
end

function [result, cache] = localBuildProductGrabFeaturesCached(trialMetadata, grabEvents, searchIntervals, reachIntervals, tracker, cache)
[searchMap, searchWarnings] = localIntervalMap(searchIntervals, "qualifying_grab_start_seconds");
[reachMap, reachWarnings] = localIntervalMap(reachIntervals, "grab_start_seconds");
if isempty(grabEvents)
    result = struct("table", localEmptyTable(localProductGrabColumns()), ...
        "warnings", localUniqueWarnings([string(searchWarnings(:)); string(reachWarnings(:))]));
    return;
end
warnings = [string(searchWarnings(:)); string(reachWarnings(:))];
[~, order] = sortrows([ [grabEvents.grab_start_seconds]', double(string({grabEvents.hand}') == "right") ], [1 2]);
ordered = grabEvents(order);
rows = repmat(localProductRowTemplate(), 0, 1);
metadataFields = ["participant_id", "session_id", "source_tracker_csv_filename", ...
    "participant_group", "condition_name", "difficulty_level", "trial_order", "language"];
for index = 1:numel(ordered)
    event = ordered(index);
    row = localProductRowTemplate();
    for field = metadataFields
        row.(char(field)) = localRecordValue(trialMetadata, field);
    end
    row.canonical_product_name = string(event.canonical_product_name);
    row.raw_product_label = string(event.raw_product_label);
    row.hand = string(event.hand);
    row.grab_start_seconds = event.grab_start_seconds;
    row.grab_release_seconds = event.grab_release_seconds;
    row.grab_duration_seconds = event.grab_duration_seconds;
    row.is_on_list = event.is_on_list;
    row.is_first_time_on_list = event.is_first_time_on_list;
    row.is_repeated_on_list = event.is_on_list && ~event.is_first_time_on_list;
    row.is_off_list = ~event.is_on_list;
    row.overlaps_other_hand_grab = event.overlaps_other_hand_grab;
    eventWarnings = strings(0, 1);
    key = localEventKey(event.canonical_product_name, event.grab_start_seconds);
    if event.is_first_time_on_list && ~isempty(key) && isKey(searchMap, key)
        search = searchMap(key);
        row.search_start_seconds = search.search_start_seconds;
        row.first_target_focus_seconds = search.first_target_focus_seconds;
        row.search_interval_valid = search.is_valid;
    elseif event.is_first_time_on_list
        eventWarnings(end+1) = "search_interval_unmatched:" + event.canonical_product_name + ":" + string(event.grab_start_seconds); %#ok<AGROW>
        warnings = localAddWarning(warnings, eventWarnings(end));
    end
    if event.is_first_time_on_list && ~isempty(key) && isKey(reachMap, key)
        reach = reachMap(key);
        row.reach_start_seconds = reach.reach_start_seconds;
        row.reach_interval_valid = reach.is_valid;
        if reach.is_valid && isfinite(reach.reach_start_seconds) && isfinite(event.grab_start_seconds)
            duration = event.grab_start_seconds - reach.reach_start_seconds;
            if isfinite(duration) && duration >= 0
                row.reach_duration_seconds = duration;
        [motion, cache] = localCachedHandMotion(tracker, event.hand, cache);
        motionWarnings = string(motion.warnings(:));
        eventWarnings = [eventWarnings; motionWarnings]; %#ok<AGROW>
        warnings = [warnings; motionWarnings]; %#ok<AGROW>
        [row.reach_path_ratio, pathWarnings] = localReachPathRatio(motion, ...
            reach.reach_start_seconds, event.grab_start_seconds, 0.02, ...
            event.hand, strings(0, 1));
        eventWarnings = [eventWarnings; pathWarnings(:)]; %#ok<AGROW>
        warnings = [warnings; pathWarnings(:)]; %#ok<AGROW>
            else
                eventWarnings(end+1) = "negative_reach_duration:" + event.hand; %#ok<AGROW>
            end
        end
    elseif event.is_first_time_on_list
        eventWarnings(end+1) = "reach_interval_unmatched:" + event.canonical_product_name + ":" + string(event.grab_start_seconds); %#ok<AGROW>
        warnings = localAddWarning(warnings, eventWarnings(end));
    end
    eventWarnings = localUniqueWarnings(eventWarnings);
    row.processing_warnings = strjoin(eventWarnings, ";");
    rows(end+1) = row; %#ok<AGROW>
end
result = struct("table", localRowsToTable(rows, localProductGrabColumns()), ...
    "warnings", localUniqueWarnings(warnings));
end

function row = localProductRowTemplate()
row = struct("participant_id", "", "session_id", "", ...
    "source_tracker_csv_filename", "", "participant_group", "", ...
    "condition_name", "", "difficulty_level", NaN, "trial_order", "", ...
    "language", "", "canonical_product_name", "", "raw_product_label", "", ...
    "hand", "", "grab_start_seconds", NaN, "grab_release_seconds", NaN, ...
    "grab_duration_seconds", NaN, "is_on_list", false, "is_first_time_on_list", false, ...
    "is_repeated_on_list", false, "is_off_list", false, ...
    "overlaps_other_hand_grab", false, "search_start_seconds", NaN, ...
    "first_target_focus_seconds", NaN, "search_interval_valid", NaN, ...
    "reach_start_seconds", NaN, "reach_duration_seconds", NaN, ...
    "reach_path_ratio", NaN, "reach_interval_valid", NaN, "processing_warnings", "");
end

function result = localBuildTrialFeatures(varargin)
trialMetadata = varargin{1};
pace = localOptionalArg(varargin, 2);
locating = localOptionalArg(varargin, 3);
reach = localOptionalArg(varargin, 4);
errorOutcome = localOptionalArg(varargin, 5);
validity = localOptionalArg(varargin, 6);
warnings = strings(0, 1);
if ~isempty(pace) && isfield(pace, "warnings")
    warnings = [warnings; string(pace.warnings(:))];
end
if ~isempty(locating) && isfield(locating, "warnings")
    warnings = [warnings; string(locating.warnings(:))];
end
if ~isempty(reach) && isfield(reach, "warnings")
    warnings = [warnings; string(reach.warnings(:))];
end
row = localTrialRowTemplate();
metadataFields = ["participant_id", "session_id", "source_tracker_csv_filename", ...
    "participant_group", "condition_name", "difficulty_level", "trial_order", "language"];
for field = metadataFields
    row.(char(field)) = localRecordValue(trialMetadata, field);
end
row.performance = localRecordValue(trialMetadata, "performance");
row.correct_products_collected_count = localRecordValue(trialMetadata, "correct_products_collected_count");
row.performance_percent = localRecordValue(trialMetadata, "performance_percent");
row.performance_change_from_d0_percentage_points = localRecordValue(trialMetadata, "performance_change_from_d0_percentage_points");
row.mental_demand_score_0_to_10 = localRecordValue(trialMetadata, "mental_demand_score_0_to_10");
errorFields = ["errors_missing", "errors_wrong_order", "errors_duplicate", "errors_not_in_list", ...
    "total_error_count", "error_change_from_d0"];
for field = errorFields
    if ~isempty(errorOutcome) && isfield(errorOutcome, char(field))
        row.(char(field)) = errorOutcome.(char(field));
    end
end
if ~isempty(pace)
    row.median_time_between_qualifying_grabs_seconds = localFieldOrNaN(pace, "median_time_between_qualifying_grabs_seconds");
    row.list_recheck_count = localFieldOrNaN(pace, "list_recheck_count");
    row.total_list_recheck_duration_seconds = localFieldOrNaN(pace, "total_list_recheck_duration_seconds");
end
if ~isempty(locating)
    row.median_time_to_target_seconds = localFieldOrNaN(locating, "median_time_to_target_seconds");
    row.median_irrelevant_focus_duration_seconds = localFieldOrNaN(locating, "median_irrelevant_focus_duration_seconds");
    row.median_head_turning_degrees = localFieldOrNaN(locating, "median_head_turning_degrees");
end
if ~isempty(reach)
    row.median_reach_duration_seconds = localFieldOrNaN(reach, "median_reach_duration_seconds");
    row.median_reach_path_ratio = localFieldOrNaN(reach, "median_reach_path_ratio");
end
countFields = ["real_product_grab_count", "first_time_on_list_grab_count", "list_visit_count", ...
    "valid_search_interval_count", "valid_reach_duration_count", "valid_reach_path_ratio_count"];
for field = countFields
    if ~isempty(validity)
        value = localRecordValue(validity, field);
        if ~isempty(value) && ~ismissing(string(value))
            row.(char(field)) = localNumericScalar(value);
        end
    end
end
row.processing_warnings = strjoin(localUniqueWarnings(warnings), ";");
result = struct("table", localRowsToTable(row, localTrialFeatureColumns()), ...
    "warnings", localUniqueWarnings(warnings));
end

function value = localOptionalArg(values, index)
if numel(values) < index || isempty(values{index})
    value = [];
else
    value = values{index};
end
end

function value = localFieldOrNaN(record, field)
if isstruct(record) && isfield(record, field)
    value = record.(field);
else
    value = NaN;
end
if isempty(value)
    value = NaN;
end
end

function row = localTrialRowTemplate()
row = struct("participant_id", "", "session_id", "", ...
    "source_tracker_csv_filename", "", "participant_group", "", ...
    "condition_name", "", "difficulty_level", NaN, "trial_order", "", ...
    "language", "", "performance", NaN, "correct_products_collected_count", NaN, ...
    "performance_percent", NaN, "performance_change_from_d0_percentage_points", NaN, ...
    "mental_demand_score_0_to_10", NaN, "errors_missing", NaN, ...
    "errors_wrong_order", NaN, "errors_duplicate", NaN, "errors_not_in_list", NaN, ...
    "total_error_count", NaN, "error_change_from_d0", NaN, ...
    "median_time_between_qualifying_grabs_seconds", NaN, "list_recheck_count", NaN, ...
    "total_list_recheck_duration_seconds", NaN, "median_time_to_target_seconds", NaN, ...
    "median_irrelevant_focus_duration_seconds", NaN, "median_head_turning_degrees", NaN, ...
    "median_reach_duration_seconds", NaN, "median_reach_path_ratio", NaN, ...
    "real_product_grab_count", NaN, "first_time_on_list_grab_count", NaN, ...
    "list_visit_count", NaN, "valid_search_interval_count", NaN, ...
    "valid_reach_duration_count", NaN, "valid_reach_path_ratio_count", NaN, ...
    "processing_warnings", "");
end

function outputTable = localRowsToTable(rows, columns)
if isempty(rows)
    outputTable = localEmptyTable(columns);
    return;
end
if numel(rows) > 1
    outputTable = struct2table(rows);
else
    outputTable = struct2table(rows);
end
for column = columns
    if ~ismember(column, string(outputTable.Properties.VariableNames))
        outputTable.(char(column)) = NaN(height(outputTable), 1);
    end
end
outputTable = outputTable(:, cellstr(columns));
end

function outputTable = localEmptyTable(columns)
outputTable = table();
booleanColumns = ["is_on_list", "is_first_time_on_list", "is_repeated_on_list", ...
    "is_off_list", "overlaps_other_hand_grab", "search_interval_valid", ...
    "reach_interval_valid", "tracker_file_exists", ...
    "participant_details_file_exists", "tracker_loaded", ...
    "malformed_time_header_repaired", "mental_demand_available"];
for column = columns
    if any(contains(column, ["_id", "filename", "group", "condition", "order", "language", "product", "label", "hand", "warnings", "path"]))
        outputTable.(char(column)) = strings(0, 1);
    elseif any(column == booleanColumns) || startsWith(column, "is_")
        outputTable.(char(column)) = false(0, 1);
    else
        outputTable.(char(column)) = NaN(0, 1);
    end
end
outputTable = outputTable(:, cellstr(columns));
end

function columns = localProductGrabColumns()
columns = ["participant_id", "session_id", "source_tracker_csv_filename", ...
    "participant_group", "condition_name", "difficulty_level", "trial_order", ...
    "language", "canonical_product_name", "raw_product_label", "hand", ...
    "grab_start_seconds", "grab_release_seconds", "grab_duration_seconds", ...
    "is_on_list", "is_first_time_on_list", "is_repeated_on_list", "is_off_list", ...
    "overlaps_other_hand_grab", "search_start_seconds", "first_target_focus_seconds", ...
    "search_interval_valid", "reach_start_seconds", "reach_duration_seconds", ...
    "reach_path_ratio", "reach_interval_valid", "processing_warnings"];
end

function columns = localTrialFeatureColumns()
columns = ["participant_id", "session_id", "source_tracker_csv_filename", ...
    "participant_group", "condition_name", "difficulty_level", "trial_order", ...
    "language", "performance", "correct_products_collected_count", ...
    "performance_percent", "performance_change_from_d0_percentage_points", ...
    "mental_demand_score_0_to_10", "errors_missing", "errors_wrong_order", ...
    "errors_duplicate", "errors_not_in_list", "total_error_count", ...
    "error_change_from_d0", "median_time_between_qualifying_grabs_seconds", ...
    "list_recheck_count", "total_list_recheck_duration_seconds", ...
    "median_time_to_target_seconds", "median_irrelevant_focus_duration_seconds", ...
    "median_head_turning_degrees", "median_reach_duration_seconds", ...
    "median_reach_path_ratio", "real_product_grab_count", ...
    "first_time_on_list_grab_count", "list_visit_count", ...
    "valid_search_interval_count", "valid_reach_duration_count", ...
    "valid_reach_path_ratio_count", "processing_warnings"];
end

function columns = localQcColumns()
columns = ["participant_id", "session_id", "source_tracker_csv_filename", ...
    "participant_group", "condition_name", "difficulty_level", "trial_order", ...
    "language", "tracker_path", "participant_details_path", "tracker_file_exists", ...
    "participant_details_file_exists", "tracker_loaded", "tracker_row_count", ...
    "finite_time_row_count", "participant_hand", "malformed_time_header_repaired", ...
    "processing_status", "real_product_grab_count", "first_time_on_list_grab_count", ...
    "list_visit_count", "valid_search_interval_count", "valid_reach_duration_count", ...
    "valid_reach_path_ratio_count", "missing_behavioural_measure_count", ...
    "mental_demand_available", "warning_count", "processing_warnings"];
end

function localWriteFixedTable(tableData, path, columns)
path = string(path);
[parent, ~, ~] = fileparts(path);
if strlength(parent) > 0 && ~isfolder(parent)
    mkdir(parent);
end
if isempty(tableData)
    tableData = localEmptyTable(columns);
end
for column = columns
    if ~ismember(column, string(tableData.Properties.VariableNames))
        tableData.(char(column)) = NaN(height(tableData), 1);
    end
end
tableData = tableData(:, cellstr(columns));
cellData = cell(height(tableData) + 1, numel(columns));
cellData(1, :) = cellstr(columns);
for columnIndex = 1:numel(columns)
    values = tableData{:, columnIndex};
    for rowIndex = 1:height(tableData)
        cellData{rowIndex + 1, columnIndex} = localCsvScalar(values(rowIndex));
    end
end
writecell(cellData, path, "Delimiter", ",", "QuoteStrings", true, "Encoding", "UTF-8");
end

function value = localCsvScalar(raw)
while iscell(raw)
    if isempty(raw)
        value = "";
        return;
    end
    raw = raw{1};
end
if islogical(raw)
    if ismissing(raw)
        value = "";
    elseif raw
        value = "True";
    else
        value = "False";
    end
    return;
end
if isnumeric(raw)
    if isempty(raw) || ~isfinite(double(raw))
        value = "";
    else
        value = string(sprintf("%.15g", double(raw)));
    end
    return;
end
text = string(raw);
if isempty(text) || any(ismissing(text)) || strlength(text) == 0
    value = "";
else
    value = text;
end
end

function result = localWriteDescriptivePlots(trialFeatures, outputDir)
warnings = strings(0, 1);
[frame, warnings] = localPreparePlotFrame(trialFeatures, warnings);
[relativeValues, relativeWarnings] = localRelativePerformance(frame);
warnings = [warnings; relativeWarnings(:)]; %#ok<AGROW>
frame.__relative_performance_change = relativeValues;
destination = fullfile(string(outputDir), "plots");
if ~isfolder(destination)
    mkdir(destination);
end
specs = {
    "01_time_between_correct_grabs.png", "median_time_between_qualifying_grabs_seconds", "Time between qualifying correct-product grabs (seconds)", "Time between qualifying correct-product grabs", false, false, [];
    "02_list_recheck_count.png", "list_recheck_count", "List rechecks (count)", "List recheck count", false, false, [];
    "03_total_list_recheck_time.png", "total_list_recheck_duration_seconds", "Total list recheck duration (seconds)", "Total list recheck duration", false, false, [];
    "04_time_to_locate_target.png", "median_time_to_target_seconds", "Time to locate target (seconds)", "Time to locate target", false, false, [];
    "05_irrelevant_focus_time.png", "median_irrelevant_focus_duration_seconds", "Irrelevant focus duration (seconds)", "Irrelevant focus duration", false, false, [];
    "06_reach_time.png", "median_reach_duration_seconds", "Reach duration (seconds)", "Reach duration", false, false, [];
    "07_reach_path_ratio.png", "median_reach_path_ratio", "Reach path ratio (dimensionless)", "Reach path ratio", false, false, [];
    "08_head_turning.png", "median_head_turning_degrees", "Headset angular movement (degrees)", "Headset angular movement", false, false, [];
    "09_total_errors.png", "total_error_count", "Total errors (count)", "Total errors", false, false, [];
    "10_error_change_from_D0.png", "error_change_from_d0", "Error change from D0 (count)", "Error change from D0", true, true, [];
    "11_relative_performance_change.png", "__relative_performance_change", "Relative performance change (percentage points)", "Relative performance change", true, true, [];
    "12_subjective_mental_demand.png", "mental_demand_score_0_to_10", "Subjective mental demand (0-10)", "Subjective mental demand", false, false, [0, 10];
    };
paths = strings(13, 1);
for index = 1:size(specs, 1)
    filename = string(specs{index, 1});
    metric = string(specs{index, 2});
    metricColumn = "__plot_" + metric;
    if metric == "__relative_performance_change"
        metricColumn = "__relative_performance_change";
    end
    path = fullfile(destination, filename);
    localDrawMetricPlot(frame, metricColumn, filename, string(specs{index, 3}), ...
        string(specs{index, 4}), path, logical(specs{index, 5}), logical(specs{index, 6}), specs{index, 7});
    paths(index) = string(path);
end
coveragePath = fullfile(destination, "13_valid_measurements_by_difficulty.png");
localDrawCoveragePlot(frame, coveragePath);
paths(13) = string(coveragePath);
result = struct("paths", paths, "warnings", localUniqueWarnings(warnings));
end

function [frame, warnings] = localPreparePlotFrame(trialFeatures, warnings)
if ~istable(trialFeatures)
    error("fe01:InvalidInput", "Trial features must be a table.");
end
frame = trialFeatures;
identifiers = ["participant_id", "session_id", "source_tracker_csv_filename", ...
    "participant_group", "condition_name", "difficulty_level", "trial_order", "language"];
for column = identifiers
    if ~ismember(column, string(frame.Properties.VariableNames))
        warnings(end+1) = "missing required identifier column: " + column; %#ok<AGROW>
        frame.(char(column)) = strings(height(frame), 1);
    end
    values = string(frame.(char(column)));
    if any(ismissing(values) | strlength(strtrim(values)) == 0)
        warnings(end+1) = "missing identifier value: " + column; %#ok<AGROW>
    end
end
conditions = strings(height(frame), 1);
for index = 1:height(frame)
    raw = lower(strtrim(string(frame.condition_name(index))));
    if raw == "visual"
        conditions(index) = "Visual";
    elseif raw == "auditory"
        conditions(index) = "Auditory";
    elseif raw == "cognitive"
        conditions(index) = "Cognitive";
    elseif ~ismissing(raw) && strlength(raw) > 0
        warnings(end+1) = "unknown condition: " + raw; %#ok<AGROW>
    end
end
frame.__plot_condition = conditions;
difficulties = NaN(height(frame), 1);
for index = 1:height(frame)
    raw = strtrim(string(frame.difficulty_level(index)));
    if startsWith(lower(raw), "d")
        raw = extractAfter(raw, 1);
    end
    value = str2double(raw);
    if isfinite(value) && any(value == [0, 2, 6, 10])
        difficulties(index) = value;
    elseif strlength(raw) > 0 && ~ismissing(raw)
        warnings(end+1) = "unknown difficulty: " + raw; %#ok<AGROW>
    end
end
frame.__plot_difficulty = difficulties;
participants = strings(height(frame), 1);
for index = 1:height(frame)
    try
        participants(index) = string(localNormalizeParticipant(frame.participant_id(index)));
    catch
        participants(index) = lower(strtrim(string(frame.participant_id(index))));
    end
end
frame.__plot_participant = participants;
% Current tables expose performance_percent. Legacy review tables may only
% contain the raw performance field; support that form without reporting a
% spurious missing performance_percent warning.
names = string(frame.Properties.VariableNames);
if ismember("performance_percent", names)
    [frame.__plot_performance_percent, warnings] = localPlotNumeric( ...
        frame, "performance_percent", warnings);
else
    frame.__plot_performance_percent = NaN(height(frame), 1);
end
if ismember("performance", names)
    [frame.__plot_performance, warnings] = localPlotNumeric( ...
        frame, "performance", warnings);
elseif ~ismember("performance_percent", names)
    [frame.__plot_performance, warnings] = localPlotNumeric( ...
        frame, "performance", warnings);
else
    frame.__plot_performance = NaN(height(frame), 1);
end
metrics = ["mental_demand_score_0_to_10", ...
    "total_error_count", "error_change_from_d0", ...
    "median_time_between_qualifying_grabs_seconds", "list_recheck_count", ...
    "total_list_recheck_duration_seconds", "median_time_to_target_seconds", ...
    "median_irrelevant_focus_duration_seconds", "median_head_turning_degrees", ...
    "median_reach_duration_seconds", "median_reach_path_ratio"];
for metric = metrics
    [frame.(char("__plot_" + metric)), warnings] = localPlotNumeric(frame, metric, warnings);
end
end

function [values, warnings] = localPlotNumeric(frame, metric, warnings)
values = NaN(height(frame), 1);
if ~ismember(metric, string(frame.Properties.VariableNames))
    warnings(end+1) = "missing metric column: " + metric; %#ok<AGROW>
    return;
end
raw = frame.(char(metric));
values = localNumericVector(raw);
present = ~localAllMissing(raw);
if any(present & ~isfinite(values))
    warnings(end+1) = "invalid numeric values: " + metric; %#ok<AGROW>
end
values(~isfinite(values)) = NaN;
end

function [changes, warnings] = localRelativePerformance(frame)
warnings = strings(0, 1);
if ismember("__plot_performance_percent", string(frame.Properties.VariableNames)) ...
        && any(isfinite(frame.__plot_performance_percent))
    performance = frame.__plot_performance_percent;
else
    performance = frame.__plot_performance;
end
keys = strings(0, 1);
values = cell(0, 1);
for index = 1:height(frame)
    if frame.__plot_difficulty(index) == 0 && strlength(frame.__plot_participant(index)) > 0 ...
            && strlength(frame.__plot_condition(index)) > 0 && isfinite(performance(index))
        key = frame.__plot_participant(index) + "|" + frame.__plot_condition(index);
        position = find(keys == key, 1, "first");
        if isempty(position)
            keys(end+1) = key; %#ok<AGROW>
            values{end+1} = performance(index); %#ok<AGROW>
        else
            values{position}(end+1) = performance(index);
        end
    end
end
baselines = containers.Map("KeyType", "char", "ValueType", "double");
for index = 1:numel(keys)
    if numel(values{index}) > 1
        parts = split(keys(index), "|");
        warnings = localAddWarning(warnings, "duplicate D0 performance baseline: " + parts(1) + "/" + parts(2));
    end
    baselines(char(keys(index))) = median(values{index});
end
changes = NaN(height(frame), 1);
missingKeys = strings(0, 1);
for index = 1:height(frame)
    difficulty = frame.__plot_difficulty(index);
    if ~isfinite(difficulty) || difficulty == 0
        continue;
    end
    key = frame.__plot_participant(index) + "|" + frame.__plot_condition(index);
    if isKey(baselines, char(key)) && isfinite(performance(index))
        changes(index) = baselines(char(key)) - performance(index);
    elseif ~any(missingKeys == key)
        missingKeys(end+1) = key; %#ok<AGROW>
    end
end
for index = 1:numel(missingKeys)
    parts = split(missingKeys(index), "|");
    participant = parts(1);
    condition = parts(2);
    if strlength(participant) == 0
        participant = "missing participant_id";
    end
    if strlength(condition) == 0
        condition = "missing condition_name";
    end
    warnings = localAddWarning(warnings, "performance baseline unavailable: " + participant + "/" + condition);
end
end

function localDrawMetricPlot(frame, metricColumn, filename, ylabel, titleText, path, excludeD0, zeroLine, yLimits)
figureHandle = figure("Visible", "off", "Color", "white");
layout = tiledlayout(figureHandle, 1, 3, "TileSpacing", "compact", "Padding", "compact");
conditions = ["Visual", "Auditory", "Cognitive"];
difficulties = [0, 2, 6, 10];
for conditionIndex = 1:3
    axisHandle = nexttile(layout, conditionIndex);
    hold(axisHandle, "on");
    medianX = NaN(0, 1);
    medianY = NaN(0, 1);
    for difficultyIndex = 1:4
        difficulty = difficulties(difficultyIndex);
        if excludeD0 && difficulty == 0
            continue;
        end
        mask = frame.__plot_condition == conditions(conditionIndex) ...
            & frame.__plot_difficulty == difficulty;
        values = frame.(char(metricColumn));
        values = values(mask);
        values = values(isfinite(values));
        if isempty(values)
            continue;
        end
        jitter = linspace(-0.08, 0.08, numel(values))';
        scatter(axisHandle, repmat(difficultyIndex, numel(values), 1) + jitter, values, ...
            22, "MarkerFaceColor", [0.54, 0.64, 0.78], "MarkerEdgeColor", "none", ...
            "DisplayName", ternary(difficultyIndex == 1, "Trial observations", ""));
        medianX(end+1) = difficultyIndex; %#ok<AGROW>
        medianY(end+1) = median(values); %#ok<AGROW>
    end
    if ~isempty(medianX)
        plot(axisHandle, medianX, medianY, "-o", "Color", [0.77, 0.31, 0.32], ...
            "LineWidth", 1.8, "DisplayName", "Condition/difficulty median");
    else
        text(axisHandle, 0.5, 0.5, "No finite measurements", "Units", "normalized", ...
            "HorizontalAlignment", "center");
    end
    if zeroLine
        yline(axisHandle, 0, "--", "Color", [0.33, 0.33, 0.33], "DisplayName", "Zero reference");
    end
    title(axisHandle, conditions(conditionIndex));
    xticks(axisHandle, 1:4);
    xticklabels(axisHandle, ["D0", "D2", "D6", "D10"]);
    xlabel(axisHandle, "Difficulty");
    ylabel(axisHandle, ylabel);
    if ~isempty(yLimits)
        ylim(axisHandle, yLimits);
    end
    grid(axisHandle, "on");
    hold(axisHandle, "off");
end
sgtitle(layout, titleText);
legend(layout, "Location", "southoutside", "Orientation", "horizontal");
exportgraphics(figureHandle, path, "Resolution", 120);
close(figureHandle);
end

function localDrawCoveragePlot(frame, path)
figureHandle = figure("Visible", "off", "Color", "white");
layout = tiledlayout(figureHandle, 1, 3, "TileSpacing", "compact", "Padding", "compact");
conditions = ["Visual", "Auditory", "Cognitive"];
difficulties = [0, 2, 6, 10];
features = ["median_time_between_qualifying_grabs_seconds", "list_recheck_count", ...
    "total_list_recheck_duration_seconds", "median_time_to_target_seconds", ...
    "median_irrelevant_focus_duration_seconds", "median_head_turning_degrees", ...
    "median_reach_duration_seconds", "median_reach_path_ratio"];
for conditionIndex = 1:3
    axisHandle = nexttile(layout, conditionIndex);
    hold(axisHandle, "on");
    for featureIndex = 1:numel(features)
        metricColumn = "__plot_" + features(featureIndex);
        percentages = NaN(1, 4);
        for difficultyIndex = 1:4
            mask = frame.__plot_condition == conditions(conditionIndex) ...
                & frame.__plot_difficulty == difficulties(difficultyIndex);
            denominator = sum(mask);
            if denominator > 0
                values = frame.(char(metricColumn));
                percentages(difficultyIndex) = 100 * sum(isfinite(values(mask))) / denominator;
            end
        end
        if any(isfinite(percentages))
            plot(axisHandle, 1:4, percentages, "-o", "LineWidth", 1.2, ...
                "DisplayName", strrep(features(featureIndex), "_", " "));
        end
    end
    title(axisHandle, conditions(conditionIndex));
    xticks(axisHandle, 1:4);
    xticklabels(axisHandle, ["D0", "D2", "D6", "D10"]);
    xlabel(axisHandle, "Difficulty");
    ylabel(axisHandle, "Valid measurements (%)");
    ylim(axisHandle, [0, 100]);
    grid(axisHandle, "on");
    hold(axisHandle, "off");
end
sgtitle(layout, "Valid measurements by feature, condition, and difficulty");
legend(layout, "Location", "southoutside", "Orientation", "horizontal");
exportgraphics(figureHandle, path, "Resolution", 120);
close(figureHandle);
end

function value = ternary(condition, whenTrue, whenFalse)
if condition
    value = whenTrue;
else
    value = whenFalse;
end
end

function [values, warnings, available] = localReadMentalDemand(path)
path = string(path);
values = containers.Map("KeyType", "char", "ValueType", "any");
warnings = strings(0, 1);
available = false;
if ~isfile(path)
    warnings(end+1) = "mental_demand_source_unavailable";
    return;
end
try
    tableData = readtable(path, "Sheet", "MD - Mental Demand", "TextType", "string", ...
        "VariableNamingRule", "preserve");
catch
    try
        tableData = readtable(path, "TextType", "string", "VariableNamingRule", "preserve");
    catch exception
        warnings(end+1) = "mental_demand_read_error:" + string(exception.identifier);
        return;
    end
end
aliases = containers.Map( ...
    {"participant", "participant_id", "group", "condition", "trial", "trial_order", "level", "difficulty", "md", "mental_demand", "mental_demand_score_0_to_10"}, ...
    {"participant", "participant", "group", "condition", "trial_order", "trial_order", "difficulty", "difficulty", "mental_demand", "mental_demand", "mental_demand"});
names = string(tableData.Properties.VariableNames);
for index = 1:numel(names)
    key = lower(strtrim(names(index)));
    if isKey(aliases, char(key))
        tableData.Properties.VariableNames{index} = aliases(char(key));
    end
end
required = ["participant", "condition", "trial_order", "difficulty", "mental_demand"];
names = string(tableData.Properties.VariableNames);
missing = required(~ismember(required, names));
if ~isempty(missing)
    warnings(end+1) = "mental_demand_columns_missing:" + strjoin(missing, ",");
    return;
end
hasGroup = ismember("group", names);
for rowIndex = 1:height(tableData)
    try
        participant = localNormalizeParticipant(tableData.participant(rowIndex));
        condition = localNormalizeCondition(tableData.condition(rowIndex));
        trialOrder = localNormalizeTrialOrder(tableData.trial_order(rowIndex));
        difficulty = localNormalizeDifficulty(tableData.difficulty(rowIndex));
    catch
        warnings(end+1) = "mental_demand_invalid_identifier_row:" + string(rowIndex + 1); %#ok<AGROW>
        continue;
    end
    if hasGroup
        group = lower(strtrim(string(tableData.group(rowIndex))));
    else
        group = "";
    end
    score = localNumericScalar(tableData.mental_demand(rowIndex));
    if ~isfinite(score)
        warnings(end+1) = "mental_demand_invalid_value_row:" + string(rowIndex + 1); %#ok<AGROW>
        continue;
    end
    if score < 0 || score > 10
        warnings(end+1) = "mental_demand_out_of_range_row:" + string(rowIndex + 1); %#ok<AGROW>
        continue;
    end
    key = localMentalDemandKey(participant, group, condition, trialOrder, difficulty);
    if isKey(values, key)
        existing = values(key);
        existing(end+1) = score;
        values(key) = existing;
    else
        values(key) = score;
    end
end
available = true;
warnings = localUniqueWarnings(warnings);
end

function key = localMentalDemandKey(participant, group, condition, trialOrder, difficulty)
key = sprintf("%s|%s|%s|%s|%g", char(participant), char(group), char(condition), char(trialOrder), double(difficulty));
end

function [value, warning] = localMentalDemandForTrial(row, values, available)
warning = strings(0, 1);
value = NaN;
if ~available
    warning = "mental_demand_source_unavailable";
    return;
end
participant = string(localRecordValue(row, "participant_id"));
group = lower(strtrim(string(localRecordValue(row, "participant_group"))));
condition = string(localRecordValue(row, "condition_name"));
trialOrder = string(localRecordValue(row, "trial_order"));
difficulty = localNumericScalar(localRecordValue(row, "difficulty_level"));
key = localMentalDemandKey(participant, group, condition, trialOrder, difficulty);
if ~isKey(values, key)
    warning = "mental_demand_match_missing:" + participant + ":" + condition + ":" + trialOrder + ":D" + string(difficulty);
    return;
end
matched = values(key);
if numel(matched) > 1
    value = median(matched);
    warning = "duplicate_mental_demand_match:" + participant + ":" + condition + ":" + trialOrder + ":D" + string(difficulty);
else
    value = matched(1);
end
end

function result = localRunFullExtraction(performancePath, rawRoot, correctListsPath, outputDir, varargin)
limit = Inf;
mentalDemandPath = "";
for index = 1:2:numel(varargin)
    option = lower(string(varargin{index}));
    value = varargin{index+1};
    switch option
        case "limit"
            limit = double(value);
            if limit < 1
                error("fe01:InvalidLimit", "Limit must be at least 1.");
            end
        case "mentaldemandpath"
            mentalDemandPath = string(value);
        otherwise
            error("fe01:UnknownOption", "Unknown full-extraction option: %s", option);
    end
end
performancePath = string(performancePath);
rawRoot = string(rawRoot);
correctListsPath = string(correctListsPath);
outputDir = string(outputDir);
trials = localStandardizeTrialTable(localReadTable(performancePath));
sourceNames = lower(strtrim(string(trials.Properties.VariableNames)));
sourceHasExplicitPercentage = any(ismember(sourceNames, ["performance_percent", "performance_percentage", "performance_pct", "percentage_performance"]));
if isfinite(limit)
    trials = trials(1:min(height(trials), limit), :);
end
catalog = localParseCorrectLists(correctListsPath);
[trials, performanceWarnings, performanceUnits] = localDerivePerformanceMetrics(trials);
errorResult = localDeriveErrorOutcomes(trials);
if strlength(mentalDemandPath) == 0
    mentalDemandPath = fullfile(fileparts(correctListsPath), "mental_demand.xlsx");
end
[mentalValues, mentalWarnings, mentalAvailable] = localReadMentalDemand(mentalDemandPath);
runWarnings = [string(performanceWarnings(:)); string(errorResult.warnings(:)); string(mentalWarnings(:))];
errorMap = containers.Map("KeyType", "char", "ValueType", "any");
for index = 1:numel(errorResult.records)
    key = localErrorOutcomeKey(errorResult.records(index));
    if isKey(errorMap, key)
        runWarnings(end+1) = "duplicate_error_outcome:" + key; %#ok<AGROW>
    else
        errorMap(key) = errorResult.records(index);
    end
end
productTables = cell(0, 1);
trialTables = cell(0, 1);
qcRows = repmat(localQcRowTemplate(), 0, 1);
rowCount = height(trials);
for rowIndex = 1:rowCount
    sourceRow = localTableRowStruct(trials, rowIndex);
    metadata = sourceRow;
    trialWarnings = strings(0, 1);
    [mentalValue, mentalWarning] = localMentalDemandForTrial(sourceRow, mentalValues, mentalAvailable);
    metadata.mental_demand_score_0_to_10 = mentalValue;
    trialWarnings = [trialWarnings; string(mentalWarning(:))]; %#ok<AGROW>
    paths = localBuildTrialPaths(sourceRow, rawRoot);
    trackerExists = isfile(paths.tracker);
    detailsExists = isfile(paths.participant_details);
    handResult = localReadParticipantHand(paths.participant_details);
    trialWarnings = [trialWarnings; string(handResult.warnings(:))]; %#ok<AGROW>
    trackerLoaded = false;
    trackerRowCount = NaN;
    finiteTimeCount = NaN;
    repairedHeader = false;
    grabEvents = localEmptyEvents();
    searchIntervals = localEmptySearchIntervals();
    reachIntervals = localEmptyReachIntervals();
    listVisits = localEmptyListVisits();
    pace = [];
    locating = [];
    reachFeatures = [];
    validity = [];
    if ~trackerExists
        trialWarnings(end+1) = "tracker_file_missing"; %#ok<AGROW>
    else
        try
            trackerResult = localLoadTrackerTable(paths.tracker);
            tracker = trackerResult.table;
            cache = localMakeCache(tracker);
            trackerLoaded = true;
            trackerRowCount = height(tracker);
            finiteTimeCount = sum(isfinite(cache.time_seconds));
            repairedHeader = trackerResult.malformed_time_header_repaired;
            trialWarnings = [trialWarnings; string(trackerResult.warnings(:))]; %#ok<AGROW>
            [grabResult, cache] = localDetectProductGrabEventsCached(tracker, catalog, ...
                sourceRow.difficulty_level, sourceRow.language, cache);
            grabEvents = grabResult.events;
            trialWarnings = [trialWarnings; string(grabResult.warnings(:))]; %#ok<AGROW>
            [focusResult, cache] = localDetectFocusEpisodesCached(tracker, catalog, cache);
            trialWarnings = [trialWarnings; string(focusResult.warnings(:))]; %#ok<AGROW>
            listVisits = localDetectListVisits(focusResult.episodes);
            [searchResult, cache] = localDetectSearchIntervalsCached(tracker, grabEvents, ...
                listVisits, focusResult.episodes, cache);
            searchIntervals = searchResult.intervals;
            trialWarnings = [trialWarnings; string(searchResult.warnings(:))]; %#ok<AGROW>
            [reachResult, cache] = localDetectReachIntervalsCached(tracker, grabEvents, ...
                searchIntervals, 0.05, 0.05, 0.10, cache);
            reachIntervals = reachResult.intervals;
            trialWarnings = [trialWarnings; string(reachResult.warnings(:))]; %#ok<AGROW>
            pace = localAggregatePace(grabEvents, listVisits);
            [locating, cache] = localAggregateLocatingCached(searchIntervals, ...
                focusResult.episodes, tracker, catalog, cache);
            trialWarnings = [trialWarnings; string(locating.warnings(:))]; %#ok<AGROW>
            [reachFeatures, cache] = localAggregateReachCached(reachIntervals, tracker, 0.02, cache);
            trialWarnings = [trialWarnings; string(reachFeatures.warnings(:))]; %#ok<AGROW>
            [productResult, cache] = localBuildProductGrabFeaturesCached(metadata, grabEvents, ...
                searchIntervals, reachIntervals, tracker, cache);
            productTables{end+1} = productResult.table; %#ok<AGROW>
            trialWarnings = [trialWarnings; string(productResult.warnings(:))]; %#ok<AGROW>
            if ~isempty(productResult.table) && ismember("processing_warnings", string(productResult.table.Properties.VariableNames))
                cells = string(productResult.table.processing_warnings);
                for warningIndex = 1:numel(cells)
                    if strlength(cells(warningIndex)) > 0
                        trialWarnings = [trialWarnings; split(cells(warningIndex), ";")]; %#ok<AGROW>
                    end
                end
            end
            validity = struct("real_product_grab_count", numel(grabEvents), ...
                "first_time_on_list_grab_count", sum([grabEvents.is_first_time_on_list]), ...
                "list_visit_count", numel(listVisits), ...
                "valid_search_interval_count", sum([searchIntervals.is_valid]), ...
                "valid_reach_duration_count", sum(isfinite([reachIntervals.reach_start_seconds])), ...
                "valid_reach_path_ratio_count", sum(isfinite([reachIntervals.reach_start_seconds])));
            % Per-event reach ratios are calculated by the product builder; use its
            % finite output columns for the two path-related validity counts.
            if ~isempty(productResult.table)
                validity.valid_reach_duration_count = sum(isfinite(productResult.table.reach_duration_seconds));
                validity.valid_reach_path_ratio_count = sum(isfinite(productResult.table.reach_path_ratio));
            end
        catch exception
            trialWarnings(end+1) = "tracker_processing_failed:" + string(exception.identifier); %#ok<AGROW>
        end
    end
    errorKey = localErrorKeyFromRow(sourceRow);
    if isKey(errorMap, errorKey)
        errorOutcome = errorMap(errorKey);
    else
        errorOutcome = [];
        trialWarnings(end+1) = "error_outcome_unmatched"; %#ok<AGROW>
    end
    trialResult = localBuildTrialFeatures(metadata, pace, locating, reachFeatures, errorOutcome, validity);
    trialWarnings = [trialWarnings; string(trialResult.warnings(:))]; %#ok<AGROW>
    trialWarnings = localUniqueWarnings(trialWarnings);
    trialTable = trialResult.table;
    trialTable.processing_warnings(1) = strjoin(trialWarnings, ";");
    trialTables{end+1} = trialTable; %#ok<AGROW>
    missingCount = NaN;
    if trackerLoaded
        behavioural = ["median_time_between_qualifying_grabs_seconds", ...
            "list_recheck_count", "total_list_recheck_duration_seconds", ...
            "median_time_to_target_seconds", "median_irrelevant_focus_duration_seconds", ...
            "median_head_turning_degrees", "median_reach_duration_seconds", ...
            "median_reach_path_ratio"];
        missingCount = sum(ismissing(trialTable{1, cellstr(behavioural)}));
    end
    status = "warning";
    if ~trackerLoaded
        status = "failed";
    elseif isempty(trialWarnings)
        status = "ok";
    end
    qc = localQcRowTemplate();
    metadataFields = ["participant_id", "session_id", "source_tracker_csv_filename", ...
        "participant_group", "condition_name", "difficulty_level", "trial_order", "language"];
    for field = metadataFields
        qc.(char(field)) = sourceRow.(char(field));
    end
    qc.tracker_path = string(paths.tracker);
    qc.participant_details_path = string(paths.participant_details);
    qc.tracker_file_exists = trackerExists;
    qc.participant_details_file_exists = detailsExists;
    qc.tracker_loaded = trackerLoaded;
    qc.tracker_row_count = trackerRowCount;
    qc.finite_time_row_count = finiteTimeCount;
    qc.participant_hand = handResult.hand;
    qc.malformed_time_header_repaired = repairedHeader;
    qc.processing_status = status;
    if ~isempty(validity)
        qc.real_product_grab_count = validity.real_product_grab_count;
        qc.first_time_on_list_grab_count = validity.first_time_on_list_grab_count;
        qc.list_visit_count = validity.list_visit_count;
        qc.valid_search_interval_count = validity.valid_search_interval_count;
        qc.valid_reach_duration_count = validity.valid_reach_duration_count;
        qc.valid_reach_path_ratio_count = validity.valid_reach_path_ratio_count;
    end
    qc.missing_behavioural_measure_count = missingCount;
    qc.mental_demand_available = isfinite(mentalValue);
    qc.warning_count = numel(trialWarnings);
    qc.processing_warnings = strjoin(trialWarnings, ";");
    qcRows(end+1) = qc; %#ok<AGROW>
    runWarnings = [runWarnings; trialWarnings(:)]; %#ok<AGROW>
end
productTable = localConcatTables(productTables, localProductGrabColumns());
trialTable = localConcatTables(trialTables, localTrialFeatureColumns());
qcTable = localRowsToTable(qcRows, localQcColumns());
if ~isfolder(outputDir)
    mkdir(outputDir);
end
productPath = fullfile(outputDir, "product_grab_features_v2.csv");
trialPath = fullfile(outputDir, "trial_features_v2.csv");
qcPath = fullfile(outputDir, "feature_extraction_qc.csv");
manifestPath = fullfile(outputDir, "feature_extraction_manifest.json");
localWriteFixedTable(productTable, productPath, localProductGrabColumns());
localWriteFixedTable(trialTable, trialPath, localTrialFeatureColumns());
localWriteFixedTable(qcTable, qcPath, localQcColumns());
plotResult = localWriteDescriptivePlots(trialTable, outputDir);
runWarnings = localUniqueWarnings([runWarnings; string(plotResult.warnings(:))]);
outputNames = [string(productPath); string(trialPath); string(qcPath); string(manifestPath); string(plotResult.paths(:))];
localWriteManifest(manifestPath, performancePath, rawRoot, correctListsPath, mentalDemandPath, ...
    outputNames, trialTable, productTable, qcTable, runWarnings, performanceUnits, sourceHasExplicitPercentage);
result = struct("productGrabTable", productTable, "trialFeatureTable", trialTable, ...
    "qcTable", qcTable, "outputPaths", outputNames, "warnings", runWarnings);
end

function row = localTableRowStruct(tableData, rowIndex)
row = struct();
for index = 1:width(tableData)
    name = tableData.Properties.VariableNames{index};
    value = tableData{rowIndex, index};
    if iscell(value)
        value = value{1};
    end
    if isstring(value) && numel(value) > 1
        value = value(1);
    end
    row.(name) = value;
end
end

function key = localErrorOutcomeKey(record)
key = sprintf("%s|%s|%s|%g|%s", char(record.participant_id), char(record.session_id), ...
    char(record.condition_name), double(record.difficulty_level), char(record.trial_order));
end

function key = localErrorKeyFromRow(row)
key = sprintf("%s|%s|%s|%g|%s", char(row.participant_id), char(row.session_id), ...
    char(row.condition_name), double(row.difficulty_level), char(row.trial_order));
end

function rows = localEmptyEvents()
rows = struct("canonical_product_name", {}, "raw_product_label", {}, "hand", {}, ...
    "grab_start_seconds", {}, "grab_release_seconds", {}, "grab_duration_seconds", {}, ...
    "is_on_list", {}, "is_first_time_on_list", {}, "overlaps_other_hand_grab", {});
end

function rows = localEmptySearchIntervals()
rows = struct("canonical_product_name", {}, "qualifying_grab_start_seconds", {}, ...
    "search_start_seconds", {}, "first_target_focus_seconds", {}, "is_valid", {});
end

function rows = localEmptyReachIntervals()
rows = struct("canonical_product_name", {}, "hand", {}, "reach_start_seconds", {}, ...
    "grab_start_seconds", {}, "is_valid", {});
end

function rows = localEmptyListVisits()
rows = struct("list_visit_start_seconds", {}, "list_visit_end_seconds", {}, ...
    "list_visit_duration_seconds", {}, "is_initial_view", {});
end

function row = localQcRowTemplate()
row = struct("participant_id", "", "session_id", "", "source_tracker_csv_filename", "", ...
    "participant_group", "", "condition_name", "", "difficulty_level", NaN, ...
    "trial_order", "", "language", "", "tracker_path", "", ...
    "participant_details_path", "", "tracker_file_exists", false, ...
    "participant_details_file_exists", false, "tracker_loaded", false, ...
    "tracker_row_count", NaN, "finite_time_row_count", NaN, "participant_hand", "", ...
    "malformed_time_header_repaired", false, "processing_status", "", ...
    "real_product_grab_count", NaN, "first_time_on_list_grab_count", NaN, ...
    "list_visit_count", NaN, "valid_search_interval_count", NaN, ...
    "valid_reach_duration_count", NaN, "valid_reach_path_ratio_count", NaN, ...
    "missing_behavioural_measure_count", NaN, "mental_demand_available", false, ...
    "warning_count", NaN, "processing_warnings", "");
end

function outputTable = localConcatTables(tables, columns)
if isempty(tables)
    outputTable = localEmptyTable(columns);
else
    outputTable = vertcat(tables{:});
    outputTable = outputTable(:, cellstr(columns));
end
end

function localWriteManifest(path, performancePath, rawRoot, correctListsPath, ...
    mentalDemandPath, outputNames, trialTable, productTable, qcTable, ...
    runWarnings, performanceUnits, sourceHasExplicitPercentage)
%LOCALWRITEMANIFEST Write stable provenance and configuration metadata.
path = string(path);
[parent, ~, ~] = fileparts(path);
if strlength(parent) > 0 && ~isfolder(parent)
    mkdir(parent);
end

qcNames = string(qcTable.Properties.VariableNames);
failedTrials = 0;
repairedHeaders = 0;
if ismember("processing_status", qcNames)
    failedTrials = sum(string(qcTable.processing_status) == "failed");
end
if ismember("malformed_time_header_repaired", qcNames)
    repairedValues = qcTable.malformed_time_header_repaired;
    if islogical(repairedValues)
        repairedHeaders = sum(repairedValues);
    else
        repairedHeaders = sum(isfinite(double(repairedValues)) & double(repairedValues) ~= 0);
    end
end
mentalDemandCount = 0;
if ismember("mental_demand_available", qcNames)
    availableValues = qcTable.mental_demand_available;
    if islogical(availableValues)
        mentalDemandCount = sum(availableValues);
    else
        mentalDemandCount = sum(isfinite(double(availableValues)) & double(availableValues) ~= 0);
    end
end

manifestOutputs = cell(numel(outputNames), 1);
for index = 1:numel(outputNames)
    [~, name, extension] = fileparts(char(string(outputNames(index))));
    if strcmpi(extension, ".png")
        manifestOutputs{index} = ['plots/', name, extension];
    else
        manifestOutputs{index} = [name, extension];
    end
end

manifest = struct();
manifest.schema_version = "feature_extraction_v2";
manifest.extractor = struct("file", "fe01_internal.m", ...
    "language", "MATLAB", "sha256", "");
manifest.inputs = struct("performance", char(string(performancePath)), ...
    "raw_root", char(string(rawRoot)), ...
    "correct_lists", char(string(correctListsPath)), ...
    "mental_demand", char(string(mentalDemandPath)));
manifest.settings = struct( ...
    "grab_minimum_duration_seconds", 0.20, ...
    "grab_merge_gap_seconds", 0.10, ...
    "reach_absolute_speed_threshold_meters_per_second", 0.05, ...
    "reach_peak_speed_fraction", 0.05, ...
    "reach_minimum_movement_duration_seconds", 0.10, ...
    "minimum_straight_distance_meters", 0.02, ...
    "performance_units", char(string(performanceUnits)), ...
    "source_has_explicit_percentage", logical(sourceHasExplicitPercentage), ...
    "reach_onset_rule", ...
    "preceding_valid_movement_step_timestamp_clipped_to_search_start");
manifest.outputs = manifestOutputs;
manifest.counts = struct("trial_rows", height(trialTable), ...
    "product_grab_rows", height(productTable), ...
    "qc_rows", height(qcTable), "failed_trials", failedTrials, ...
    "repaired_time_headers", repairedHeaders);
manifest.mental_demand_available_count = mentalDemandCount;
manifest.warnings = cellstr(localUniqueWarnings(runWarnings));

jsonText = string(jsonencode(manifest));
writelines([jsonText; ""], path, "Encoding", "UTF-8");
end
