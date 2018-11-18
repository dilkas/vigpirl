function errs = cvx_lasterr()

% CVX_LASTERR   Cleans up the LASTERR output for use in CVX constraint errors.
%    This is an internal function intended to improve the formatting of errors
%    generated by CVX for invalid constraints. It simply strips out some of
%    MATLAB's formatting so that CVX can substitute some of its own.

errs = lasterr; 
if strncmp( 'Error using ==>', errs, 15 ),
    errs = errs(min(find(errs==10))+1:end);
    while errs(end) == 10, errs(end) = []; end
end

% Copyright 2010 Michael C. Grant and Stephen P. Boyd.
% See the file COPYING.txt for full copyright information.
% The command 'cvx_where' will show where this file is located.

