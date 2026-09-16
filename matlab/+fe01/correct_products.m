function products = correct_products(catalog, difficulty, language)
%CORRECT_PRODUCTS Return canonical products for one difficulty/language.
products = fe01.fe01_internal("correct_products", catalog, difficulty, language);
end
