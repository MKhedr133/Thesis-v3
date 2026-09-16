function tf = is_catalog_product(catalog, value)
%IS_CATALOG_PRODUCT Test whether a label is a recognized catalog product.
tf = fe01.fe01_internal("is_catalog_product", catalog, value);
end
