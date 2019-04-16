classdef ThreeStateMdp < Mdp
    methods
        function obj = ThreeStateMdp()
            mdp_data = struct('discount', 0.9, 'states', 3, 'actions', 2);
            mdp_data.sa_s(:, :, 1) = [2, 3; 1, 3; 1, 2];
            mdp_data.sa_p(1:3, 1:2, 1) = 1;
            obj = obj@Mdp(mdp_data, struct('splittable', [1; 2; 3]), {[1, 1], [3, 2]})
        end

        function covariance_and_policy_with_more_data(obj)
            % Collect data about how Sigma and policies change with more and
            % more (of the same) demonstrations
            max_demonstrations_count = 2;
            num_repeats = 2;
            data = [];
            for x = 1:max_demonstrations_count
                for y = 1:max_demonstrations_count
                    fprintf('x=%d, y=%d\n', x, y);
                    obj.demonstrations = {};
                    [obj.demonstrations{1:x}] = deal([1, 1]);
                    [obj.demonstrations{x+1:x+y}] = deal([3, 2]);
                    temp_data1 = [];
                    for i = 1:num_repeats
                        result = run_until_works(obj);
                        % The format is: first column top to bottom, then second, etc.
                        temp_data1 = [temp_data1; reshape(result.gp.B * result.gp.B', [1, 9]), reshape(result.p, [1, 6])];
                    end
                    data = [data; [x, y, mean(temp_data1, 1)]];
                end
            end
            %writematrix(data, '../mpaper/covariance_and_policy.csv');
            writematrix(data, 'covariance_and_policy.csv');
        end

        function covariances_with_more_data(obj)
            % Different kinds of covariances and their means/medians
            max_demonstrations_count = 2;
            num_repeats = 2;
            data = [];
            data2 = [];
            data3 = [];
            data4 = [];
            for x = 1:max_demonstrations_count
                for y = 1:max_demonstrations_count
                    fprintf('x=%d, y=%d\n', x, y);
                    obj.demonstrations = {};
                    [obj.demonstrations{1:x}] = deal([1, 1]);
                    [obj.demonstrations{x+1:x+y}] = deal([3, 2]);
                    temp_data1 = [];
                    temp_data2 = [];
                    for i = 1:num_repeats
                        result = run_until_works(obj);
                        [covariance, Gamma] = construct_covariance_matrices(result);
                        % The format is: first column top to bottom, then second, etc.
                        temp_data1 = [temp_data1; reshape(covariance, [1, 9])];
                        temp_data2 = [temp_data2; reshape(Gamma, [1, 9])];
                    end
                    data = [data; [x, y, mean(temp_data1, 1)]];
                    data2 = [data2; [x, y, median(temp_data1, 1)]];
                    data3 = [data3; [x, y, mean(temp_data2, 1)]];
                    data4 = [data4; [x, y, median(temp_data2, 1)]];
                end
            end
            %writematrix(data, '../mpaper/mean_covariance.csv');
            %writematrix(data2, '../mpaper/median_covariance.csv');
            %writematrix(data3, '../mpaper/mean_Gamma.csv');
            %writematrix(data4, '../mpaper/median_Gamma.csv');

            writematrix(data, 'mean_covariance.csv');
            writematrix(data2, 'median_covariance.csv');
            writematrix(data3, 'mean_Gamma.csv');
            writematrix(data4, 'median_Gamma.csv');
        end
    end
end