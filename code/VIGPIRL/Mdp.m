classdef Mdp
    properties
        mdp_data        % A struct consisting of:
                        % - discount: gamma
                        % - states: number of states, |S|
                        % - actions: number of actions, |A|
                        % - sa_s: a three-dimensional array. First dimension is
                        %   for state, second dimension is for action. In the
                        %   third dimension, we list all states that can be
                        %   reached by executing that action in that state.
                        % - sa_p: same as sa_s, except instead of listing
                        %   states, we list probabilities of transitioning to
                        %   those states.
        feature_data    % struct('splittable', X), where X is the feature matrix
        demonstrations  % a comma-separated cell array with [state, action] pairs
    end
    methods
        function obj = Mdp(mdp_data, feature_data, demonstrations)
            obj.mdp_data = mdp_data;
            obj.feature_data = feature_data;
            obj.demonstrations = demonstrations;
        end
    end
end