function result = build_product_grab_features(trialMetadata, grabEvents, searchIntervals, reachIntervals, tracker)
%BUILD_PRODUCT_GRAB_FEATURES Build the fixed product-grab output table.
result = fe01.fe01_internal("build_product_grab_features", trialMetadata, grabEvents, searchIntervals, reachIntervals, tracker);
end
