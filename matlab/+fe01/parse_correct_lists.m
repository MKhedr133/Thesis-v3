function catalog = parse_correct_lists(path)
%PARSE_CORRECT_LISTS Parse bilingual CorrectLists.txt into a product catalog.
catalog = fe01.fe01_internal("parse_correct_lists", path);
end
