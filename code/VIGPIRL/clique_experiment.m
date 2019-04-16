function clique_experiment()
    n = 10;
    num_demonstrations = 100;
    num_repeats = 2;

    mdp_data = struct('discount', 0.9, 'states', n, 'actions', n - 1);
    for i = 1:n
        mdp_data.sa_s(i, :, 1) = horzcat(1:(i - 1), (i + 1):n);
    end
    mdp_data.sa_p(1:n, 1:(n - 1), 1) = 1;
    feature_data = struct('splittable', (1:n)');

    total_values = [];
    total_groups = [];
    for i = 1:num_repeats
        [random_demonstrations, targeted_demonstrations] = generate_demonstrations(num_demonstrations, n)
        random_result = run_until_works(Mdp(mdp_data, feature_data, random_demonstrations));
        targeted_result = run_until_works(Mdp(mdp_data, feature_data, targeted_demonstrations));
        random_covariance = construct_covariance_matrices(random_result);
        targeted_covariance = construct_covariance_matrices(targeted_result);
        [values, groups] = meaningful_classes(abs(random_covariance) - abs(targeted_covariance));
        total_values = [total_values, values];
        total_groups = [total_groups, groups];
    end

    figure('Units', 'centimeters', 'Position', [0 0 12 8], 'PaperPositionMode', 'auto', 'Resize', 'off');
    boxplot(total_values, total_groups);
    ylabel('difference in covariance');
    hold on;
    refline([0 0]);
    %print('../mpaper/figures/boxplots', '-depsc2');
end

function [random_demonstrations, targeted_demonstrations] = generate_demonstrations(num_demonstrations, n)
    random_demonstrations = {};
    targeted_demonstrations = {};
    for i = 1:num_demonstrations
        random_demonstrations{i} = [randi(n), randi(n - 1)];
        targeted_demonstrations{i} = [1 + randi(n - 1), 1];
    end
end

function [values, groups] = meaningful_classes(matrix_of_differences)
    diagonal = diag(matrix_of_differences);
    the_rest = get_lower_triangle(matrix_of_differences);
    state = diagonal(1);
    other_diag = diagonal(2:end);
    last_neighbour_index = size(matrix_of_differences, 1) - 1;
    neighbours = the_rest(1:last_neighbour_index);
    non_neighbours = the_rest((last_neighbour_index + 1):end);

    values = horzcat(state, other_diag', neighbours', non_neighbours');
    groups = {};
    a = 1;
    b = length(state);
    [groups{a:b}] = deal('first state');
    a = b + 1;
    b = b + length(other_diag);
    [groups{a:b}] = deal('other states');
    a = b + 1;
    b = b + length(neighbours);
    [groups{a:b}] = deal('incident edges');
    a = b + 1;
    b = b + length(non_neighbours);
    [groups{a:b}] = deal('other edges');
end