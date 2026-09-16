function result = write_descriptive_feature_plots(trialFeatures, outputDir)
%WRITE_DESCRIPTIVE_FEATURE_PLOTS Write the thirteen FE-01 descriptive plots.
result = fe01.fe01_internal("write_descriptive_feature_plots", trialFeatures, outputDir);
end
