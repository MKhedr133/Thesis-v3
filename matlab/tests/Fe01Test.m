classdef Fe01Test < matlab.unittest.TestCase
    % FE-01 native MATLAB port tests.
    %
    % These tests use small synthetic tracker tables for detector behavior,
    % then use the approved Python CSVs for an optional parity check.

    methods (TestClassSetup)
        function addPackageToPath(testCase)
            matlabRoot = fileparts(fileparts(mfilename("fullpath")));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(matlabRoot));
        end
    end

    methods (Test)
        function testIdentifierNormalisationAndTrialPaths(testCase)
            source = table("1", "s2", "T4", "Old", "visual", 0, "t4", ...
                'VariableNames', {"participant", "session", "trial", "group", ...
                "condition", "level", "trial2"});
            normalized = fe01.standardize_trial_table(source);
            testCase.verifyEqual(normalized.participant_id, "P01");
            testCase.verifyEqual(normalized.session_id, "S2");
            testCase.verifyEqual(normalized.source_tracker_csv_filename, ...
                "data_collector_vr_sample_T004.csv");
            testCase.verifyEqual(normalized.condition_name, "Visual");
            testCase.verifyEqual(normalized.difficulty_level, 0);
            testCase.verifyEqual(normalized.trial_order, "T4");
            paths = fe01.build_trial_paths(normalized(1,:), "D:\raw");
            testCase.verifyEqual(paths.tracker, ...
                fullfile("D:\raw", "P1", "P01_HMD_Data", "P01", "S2", ...
                "trackers", "data_collector_vr_sample_T004.csv"));
        end

        function testCorrectListCatalogCanonicalisesBilingualLabels(testCase)
            file = [tempname, ".txt"];
            testCase.addTeardown(@() deleteIfPresent(file));
            writelines(["correctLists.level0.EN = {'Red cola can'};"; ...
                "correctLists.level0.NL = {'Rood blikje cola'};"; ...
                "correctLists.level2.EN = {'Red cola can'};"; ...
                "correctLists.level2.NL = {'Rood blikje cola'};"; ...
                "correctLists.level6.EN = {'Red cola can'};"; ...
                "correctLists.level6.NL = {'Rood blikje cola'};"; ...
                "correctLists.level10.EN = {'Red cola can'};"; ...
                "correctLists.level10.NL = {'Rood blikje cola'};"], file);
            catalog = fe01.parse_correct_lists(file);
            testCase.verifyEqual(fe01.canonicalize_product(catalog, ...
                "Rood blikje cola (2)"), "red cola can");
            testCase.verifyTrue(fe01.is_catalog_product(catalog, "cheese") == false);
            testCase.verifyEqual(fe01.correct_products(catalog, 0, "EN"), ...
                "red cola can");
        end

        function testMalformedTimeHeaderIsRepairedInMemory(testCase)
            file = [tempname, ".csv"];
            testCase.addTeardown(@() deleteIfPresent(file));
            source = table([1; 2], [0; 0], ...
                'VariableNames', {"483383time", "enableBlackout"});
            writetable(source, file);
            loaded = fe01.load_tracker_table(file);
            testCase.verifyTrue(ismember("time", string(loaded.table.Properties.VariableNames)));
            testCase.verifyFalse(ismember("483383time", string(loaded.table.Properties.VariableNames)));
            testCase.verifyTrue(any(loaded.warnings == "repaired_time_header:483383time"));
            testCase.verifyTrue(loaded.malformed_time_header_repaired);
        end

        function testHandMotionUsesRecordedTimeAndBreaksGaps(testCase)
            tracker = table([0; 0.5; 1.5; 2], [0; 1; 2; NaN], ...
                [0; 0; 0; 0], [0; 0; 0; 0], ...
                'VariableNames', {"time", "left_pos_x", "left_pos_y", "left_pos_z"});
            motion = fe01.prepare_hand_motion(tracker, "left");
            testCase.verifyEqual([motion.samples.timestamp_seconds], [0 0.5 1.5 2]);
            testCase.verifyEqual(motion.samples(2).step_distance_meters, 1, AbsTol=1e-10);
            testCase.verifyEqual(motion.samples(2).step_speed_meters_per_second, 2, AbsTol=1e-10);
            testCase.verifyEqual(motion.samples(3).step_speed_meters_per_second, 1, AbsTol=1e-10);
            testCase.verifyFalse(motion.samples(4).is_valid);
            testCase.verifyTrue(isnan(motion.samples(4).step_distance_meters));
        end

        function testReachOnsetUsesPrecedingMovementStepTimestamp(testCase)
            tracker = Fe01Test.syntheticMovementTracker();
            event = Fe01Test.syntheticGrabEvent("red cola can", "left", 4.0);
            search = Fe01Test.syntheticSearchInterval("red cola can", 1.0, 4.0);
            result = fe01.detect_reach_intervals(tracker, event, ...
                "SearchIntervals", search);
            testCase.verifyEqual(result.intervals(1).reach_start_seconds, 2.0, AbsTol=1e-10);
            testCase.verifyTrue(result.intervals(1).is_valid);
        end

        function testLocatingAndHeadTurningPreserveZeroIrrelevantDuration(testCase)
            tracker = Fe01Test.syntheticHeadTracker();
            focus = Fe01Test.syntheticTargetFocus();
            search = Fe01Test.syntheticSearchInterval("red cola can", 0, 3);
            locating = fe01.aggregate_trial_locating_features(search, focus, tracker, ...
                Fe01Test.syntheticCatalog());
            testCase.verifyEqual(locating.median_irrelevant_focus_duration_seconds, 0, AbsTol=1e-10);
            testCase.verifyGreaterThan(locating.median_head_turning_degrees, 0);
        end

        function testErrorOutcomeTotalsAndD0Changes(testCase)
            performance = Fe01Test.syntheticPerformanceTable();
            result = fe01.derive_error_outcomes(performance);
            testCase.verifyEqual(result.records(1).total_error_count, 2);
            testCase.verifyEqual(result.records(2).error_change_from_d0, 3);
            testCase.verifyEqual(result.records(3).error_change_from_d0, 5);
            testCase.verifyEmpty(result.records(1).error_change_from_d0);
        end

        function testFixedSchemasAndBlankWriter(testCase)
            metadata = Fe01Test.syntheticTrialMetadata();
            trial = fe01.build_trial_features(metadata, [], [], [], [], []);
            expected = fe01.trial_feature_columns();
            testCase.verifyEqual(string(trial.table.Properties.VariableNames), expected);
            file = [tempname, ".csv"];
            testCase.addTeardown(@() deleteIfPresent(file));
            fe01.write_trial_features(trial.table, file);
            text = fileread(file);
            testCase.verifyFalse(contains(text, "NaN"));
        end

        function testDescriptivePlotsProduceExactlyThirteenFiles(testCase)
            tableData = Fe01Test.syntheticTrialFeatureTable();
            outputDir = string(tempname);
            testCase.addTeardown(@() removeDirectory(outputDir));
            plotResult = fe01.write_descriptive_feature_plots(tableData, outputDir);
            testCase.verifyEqual(numel(plotResult.paths), 13);
            testCase.verifyEqual(numel(dir(fullfile(outputDir, "plots", "*.png"))), 13);
            testCase.verifyEqual(string(plotResult.paths(1).name), ...
                "01_time_between_correct_grabs.png");
            testCase.verifyEqual(string(plotResult.paths(end).name), ...
                "13_valid_measurements_by_difficulty.png");
        end

        function testFullRunnerIsolatesMissingTracker(testCase)
            [root, performancePath, listsPath, mentalDemandPath] = ...
                Fe01Test.syntheticInputFiles();
            testCase.addTeardown(@() removeDirectory(root));
            outputDir = fullfile(root, "outputs");
            result = fe01.run_full_extraction(performancePath, root, listsPath, ...
                outputDir, "MentalDemandPath", mentalDemandPath);
            testCase.verifyEqual(height(result.trialFeatureTable), 2);
            testCase.verifyEqual(height(result.qcTable), 2);
            testCase.verifyTrue(any(result.qcTable.processing_status == "failed"));
            testCase.verifyTrue(isfile(fullfile(outputDir, "feature_extraction_manifest.json")));
        end

    function testPythonParityWhenReferenceOutputsAreAvailable(testCase)
        pythonDir = fullfile(fileparts(fileparts(fileparts(mfilename("fullpath")))), ...
            "outputs", "feature_extraction_v2");
            assumeTrue(testCase, isfile(fullfile(pythonDir, "trial_features_v2.csv")), ...
                "Approved Python outputs are not available for parity test.");
            reference = readtable(fullfile(pythonDir, "trial_features_v2.csv"), ...
                "TextType", "string", "VariableNamingRule", "preserve");
        testCase.verifyEqual(string(reference.Properties.VariableNames), ...
            fe01.trial_feature_columns());
    end

    function testPublicHeadsetQuaternionPreparation(testCase)
        tracker = Fe01Test.syntheticHeadTracker();
        result = fe01.prepare_headset_rotations(tracker);
        testCase.verifySize(result.rotations, [4, 5]);
        testCase.verifyEqual(result.samples(1).timestamp_seconds, 0);
        testCase.verifyEqual(norm(result.rotations(1, 2:5)), 1, AbsTol=1e-10);
        testCase.verifyEmpty(result.warnings);

        aliasResult = fe01.prepare_headset_quaternions(tracker);
        testCase.verifyEqual(aliasResult.rotations, result.rotations, AbsTol=1e-10);
    end

    function testProductBuilderReportsDuplicateIntervalKeys(testCase)
        metadata = Fe01Test.syntheticTrialMetadata();
        event = Fe01Test.syntheticGrabEvent("red cola can", "left", 4.0);
        search = Fe01Test.syntheticSearchInterval("red cola can", 1.0, 4.0);
        search(2) = search(1);
        result = fe01.build_product_grab_features(metadata, event, search, [], ...
            Fe01Test.syntheticMovementTracker());
        testCase.verifyTrue(any(contains(string(result.warnings), ...
            "duplicate_search_interval:red cola can:4")));
    end

    function testLegacyPlotTableUsesRawPerformance(testCase)
        tableData = removevars(Fe01Test.syntheticTrialFeatureTable(), ...
            "performance_percent");
        outputDir = string(tempname);
        testCase.addTeardown(@() removeDirectory(outputDir));
        result = fe01.write_descriptive_feature_plots(tableData, outputDir);
        testCase.verifyFalse(any(contains(string(result.warnings), ...
            "missing metric column: performance_percent")));
    end
end

    methods (Static, Access=private)
        function event = syntheticGrabEvent(product, hand, startTime)
            event = struct("canonical_product_name", string(product), ...
                "raw_product_label", string(product), "hand", string(hand), ...
                "grab_start_seconds", startTime, "grab_release_seconds", startTime + 0.5, ...
                "grab_duration_seconds", 0.5, "is_on_list", true, ...
                "is_first_time_on_list", true, "overlaps_other_hand_grab", false);
        end

        function interval = syntheticSearchInterval(product, startTime, grabTime)
            interval = struct("canonical_product_name", string(product), ...
                "qualifying_grab_start_seconds", grabTime, ...
                "search_start_seconds", startTime, ...
                "first_target_focus_seconds", startTime + 1, "is_valid", true);
        end

        function tracker = syntheticMovementTracker()
            tracker = table([1; 2; 3; 4], [0; 0.01; 0.10; 0.20], ...
                zeros(4,1), zeros(4,1), ...
                'VariableNames', {"time", "left_pos_x", "left_pos_y", "left_pos_z"});
        end

        function tracker = syntheticHeadTracker()
            tracker = table((0:3)', zeros(4,1), zeros(4,1), zeros(4,1), ...
                [0; 0; 0.707106781186548; 1], ...
                [0; 0; 0; 0.707106781186548], ...
                'VariableNames', {"time", "hmd_rot_x", "hmd_rot_y", ...
                "hmd_rot_z", "hmd_rot_w", "hmd_rot_extra"});
        end

        function focus = syntheticTargetFocus()
            focus = struct("canonical_focus_name", "red cola can", ...
                "raw_focus_name", "Red cola can", "cleaned_focus_tag", "product", ...
                "raw_focus_tag", "product", "focus_start_seconds", 1, ...
                "focus_end_seconds", 2, "focus_duration_seconds", 1);
        end

        function catalog = syntheticCatalog()
            catalog = struct("ordered_lists", [], "alias_to_canonical", [], ...
                "all_canonical_products", "red cola can");
        end

        function performance = syntheticPerformanceTable()
            performance = table(["P01"; "P01"; "P01"], ...
                ["S001"; "S002"; "S003"], ...
                ["data_collector_vr_sample_T001.csv"; "data_collector_vr_sample_T001.csv"; ...
                "data_collector_vr_sample_T001.csv"], ...
                ["Old"; "Old"; "Old"], ["Visual"; "Visual"; "Visual"], ...
                [0; 2; 6], ["T1"; "T1"; "T1"], ["EN"; "EN"; "EN"], ...
                [1; 3; 5], [1; 1; 1], [1; 1; 1], [0; 0; 0], [0; 0; 0], ...
                'VariableNames', {"participant_id", "session_id", ...
                "source_tracker_csv_filename", "participant_group", "condition_name", ...
                "difficulty_level", "trial_order", "language", "performance", ...
                "errors_missing", "errors_wrong_order", "errors_duplicate", ...
                "errors_not_in_list"});
        end

        function metadata = syntheticTrialMetadata()
            metadata = struct("participant_id", "P01", "session_id", "S001", ...
                "source_tracker_csv_filename", "data_collector_vr_sample_T001.csv", ...
                "participant_group", "Old", "condition_name", "Visual", ...
                "difficulty_level", 0, "trial_order", "T1", "language", "EN", ...
                "performance", NaN, "mental_demand_score_0_to_10", NaN);
        end

        function tableData = syntheticTrialFeatureTable()
            metadata = Fe01Test.syntheticTrialMetadata();
            result = fe01.build_trial_features(metadata, [], [], [], [], []);
            tableData = result.table;
            tableData = [tableData; tableData];
            tableData.condition_name = ["Visual"; "Auditory"];
            tableData.difficulty_level = [0; 2];
            tableData.median_reach_path_ratio = [1; 1.2];
            tableData.total_error_count = [0; 2];
            tableData.performance_percent = [45; 40];
            tableData.mental_demand_score_0_to_10 = [7; 8];
        end

        function [root, performancePath, listsPath, mentalDemandPath] = syntheticInputFiles()
            root = string(tempname);
            mkdir(root);
            participantRoot = fullfile(root, "P1", "P01_HMD_Data", "P01", "S001");
            mkdir(fullfile(participantRoot, "trackers"));
            mkdir(fullfile(participantRoot, "session_info"));
            tracker = table([0; 1], false(2,1), strings(2,1), false(2,1), strings(2,1), ...
                'VariableNames', {"time", "is_grabbing_left", "grabbed_object_left", ...
                "is_grabbing_right", "grabbed_object_right"});
            writetable(tracker, fullfile(participantRoot, "trackers", ...
                "data_collector_vr_sample_T001.csv"));
            writetable(table("Left", "Ivory", "Visual", ...
                'VariableNames', {"side", "skin", "condition"}), ...
                fullfile(participantRoot, "session_info", "participant_details.csv"));
            performance = table(["P01"; "P01"], ["S001"; "S001"], ...
                ["data_collector_vr_sample_T001.csv"; "missing.csv"], ...
                ["Old"; "Old"], ["Visual"; "Visual"], [0; 2], ["T1"; "T2"], ...
                ["EN"; "EN"], [1; 2], 'VariableNames', {"participant", "session", ...
                "trial", "group", "condition", "level", "trial2", "language", "performance"});
            performancePath = fullfile(root, "performance.csv");
            writetable(performance, performancePath);
            listsPath = fullfile(root, "CorrectLists.txt");
            writelines(["correctLists.level0.EN = {'Red cola can'};"; ...
                "correctLists.level0.NL = {'Rood blikje cola'};"; ...
                "correctLists.level2.EN = {'Red cola can'};"; ...
                "correctLists.level2.NL = {'Rood blikje cola'};"; ...
                "correctLists.level6.EN = {'Red cola can'};"; ...
                "correctLists.level6.NL = {'Rood blikje cola'};"; ...
                "correctLists.level10.EN = {'Red cola can'};"; ...
                "correctLists.level10.NL = {'Rood blikje cola'};"], listsPath);
            mentalDemandPath = fullfile(root, "mental_demand.xlsx");
            md = table(["P01"; "P01"], ["Old"; "Old"], ["Visual"; "Visual"], ...
                ["T1"; "T2"], [0; 2], [7; 8], ...
                'VariableNames', {"Participant", "Group", "Condition", "Trial", "Level", "MD"});
            writetable(md, mentalDemandPath, "Sheet", "MD - Mental Demand");
        end
    end
end

function deleteIfPresent(path)
if isfile(path)
    delete(path);
end
end

function removeDirectory(path)
if isfolder(path)
    rmdir(path, "s");
end
end
