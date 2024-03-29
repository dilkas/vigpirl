% Return the lower triangle part of a mtrix
function lt = get_lower_triangle(matrix)
  mask = tril(true(size(matrix)), -1);
  lt = matrix(mask);
end
