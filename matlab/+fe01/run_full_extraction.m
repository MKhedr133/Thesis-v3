function result = run_full_extraction(performancePath, rawRoot, correctListsPath, outputDir, varargin)
%RUN_FULL_EXTRACTION Run the native MATLAB FE-01 pipeline.
result = fe01.fe01_internal("run_full_extraction", performancePath, rawRoot, correctListsPath, outputDir, varargin{:});
end
