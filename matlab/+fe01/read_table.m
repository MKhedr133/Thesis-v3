function tableData = read_table(path)
%READ_TABLE Read a comma, semicolon, or tab-delimited FE-01 table.
tableData = fe01.fe01_internal("read_table", path);
end
