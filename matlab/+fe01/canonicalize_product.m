function canonical = canonicalize_product(catalog, value)
%CANONICALIZE_PRODUCT Return the cleaned English catalog product name.
canonical = fe01.fe01_internal("canonicalize_product", catalog, value);
end
