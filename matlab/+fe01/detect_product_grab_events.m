function result = detect_product_grab_events(tracker, catalog, difficulty, language)
%DETECT_PRODUCT_GRAB_EVENTS Detect complete real-product grabs.
result = fe01.fe01_internal("detect_product_grab_events", tracker, catalog, difficulty, language);
end
