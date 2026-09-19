function tableData = standardize_trial_table(data)
%STANDARDIZE_TRIAL_TABLE Normalize the FE-01 master trial identifiers.
tableData = fe01.fe01_internal("standardize_trial_table", data);
end
