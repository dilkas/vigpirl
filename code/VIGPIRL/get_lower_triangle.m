function lt = get_lower_triangle(matrix)
  mask = tril(true(size(matrix)), -1);
  lt = matrix(mask);
end
