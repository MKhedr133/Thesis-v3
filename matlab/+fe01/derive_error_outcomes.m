function result = derive_error_outcomes(performance)
%DERIVE_ERROR_OUTCOMES Normalize raw errors and derive D0 changes.
result = fe01.fe01_internal("derive_error_outcomes", performance);
end
