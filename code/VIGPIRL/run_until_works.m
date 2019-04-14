function result = run_until_works(mdp)
    % Keep calling the algorithm until it runs without errors and warnings
    warning('');
    try
        result = vigpirlrun(struct(), mdp.mdp_data, 'linearmdp', mdp.feature_data, mdp.demonstrations);
    catch
        fprintf('Failed!\n');
        result = run_until_works(mdp);
    end
    [warning_message, ~] = lastwarn;
    if ~isempty(warning_message)
        fprintf('Failed because of a warning!\n');
        result = run_until_works(mdp);
    end
end
