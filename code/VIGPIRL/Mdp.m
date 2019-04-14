classdef Mdp
    properties
        mdp_data
        feature_data
        demonstrations
    end
    methods
        function obj = Mdp(mdp_data, feature_data, demonstrations)
            obj.mdp_data = mdp_data;
            obj.feature_data = feature_data;
            obj.demonstrations = demonstrations;
        end
    end
end