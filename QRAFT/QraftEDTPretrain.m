% QraftEDTPretrain.m
clear;
clc;

dataDir = './data/qraft_training_data';
modelDir = './model/qraft_pretrained_models_edt';
if ~exist(modelDir, 'dir')
    mkdir(modelDir);
end

files = dir(fullfile(dataDir, '*_qraft_model.csv'));

summary_backend = {};
summary_num_rows = [];
summary_num_train = [];
summary_num_test = [];
summary_num_classes = [];
summary_loss = [];
summary_model_path = {};

for fidx = 1:length(files)
    filename = files(fidx).name;
    filepath = fullfile(files(fidx).folder, filename);

    [~, backend_name, ~] = fileparts(filename);
    backend_name = erase(backend_name, '_qraft_model');

    fprintf('\n==============================\n');
    fprintf('Training backend: %s\n', backend_name);
    fprintf('Reading file: %s\n', filepath);
    fprintf('==============================\n');

    data = readtable(filepath);
    data = rmmissing(data);

    % Keep only the columns needed by the full Qraft model
    data = data(:, { ...
        'ComputerID', ...
        'CircuitWidth', ...
        'CircuitDepth', ...
        'CircuitNumU1Gates', ...
        'CircuitNumU2Gates', ...
        'CircuitNumU3Gates', ...
        'CircuitNumCXGates', ...
        'TotalUpDnErr25', ...
        'TotalUpDnErr50', ...
        'TotalUpDnErr75', ...
        'StateHammingWeight', ...
        'StateUpProb25', ...
        'StateUpProb50', ...
        'StateUpProb75', ...
        'StateUpDnErr25', ...
        'StateUpDnErr50', ...
        'StateUpDnErr75', ...
        'StateRealProb'});

    % Make sure label is integer-like categorical class
    data.StateRealProb = round(data.StateRealProb);
    data.StateRealProb(data.StateRealProb < 0) = 0;
    data.StateRealProb(data.StateRealProb > 100) = 100;

    % Keep splitting until test labels are subset of train labels
    while true
        data = data(randperm(size(data, 1)), :);

        split_idx = round(height(data) * 0.85);
        train_df = data(1:split_idx, :);
        test_df  = data(split_idx+1:end, :);

        clTrn = unique(train_df{:, 'StateRealProb'});
        clTst = unique(test_df{:, 'StateRealProb'});

        isSubset = all(ismember(clTst, clTrn));
        if isSubset == 1
            break;
        end
    end

    nmcl = length(clTrn);

    % Cost matrix exactly following your train.m logic
    mscl = zeros(nmcl);
    for i = 1:nmcl
        for j = 1:nmcl
            mscl(i, j) = ((clTrn(i) - clTrn(j))^2) * (clTrn(i) + 1);
        end
    end

    fprintf('Rows: %d | Train: %d | Test: %d | Classes: %d\n', ...
        height(data), height(train_df), height(test_df), nmcl);

    % Train the full-feature Qraft ensemble-of-decision-trees model
    cvp = cvpartition(height(train_df), 'KFold', 5);

    mdT = fitcensemble( ...
        train_df, ...
        'StateRealProb', ...
        'Cost', mscl, ...
        'OptimizeHyperparameters', 'all', ...
        'HyperparameterOptimizationOptions', struct( ...
            'Optimizer', 'bayesopt', ...
            'AcquisitionFunctionName', 'expected-improvement', ...
            'MaxObjectiveEvaluations', 50, ...
            'ShowPlots', false, ...
            'Verbose', 1, ...
            'CVPartition', cvp) ...
        );

    % Evaluate
    test_loss = loss(mdT, test_df);

    % Predict on test set
    test_df.StatePredProb = predict(mdT, test_df);

    % Save model
    model_path = fullfile(modelDir, sprintf('%s_qraft_edt.mat', backend_name));
    save(model_path, 'mdT', 'clTrn', 'mscl');

    % Save predictions
    pred_path = fullfile(modelDir, sprintf('%s_qraft_test_predictions.csv', backend_name));
    writetable(test_df, pred_path);

    fprintf('Saved model: %s\n', model_path);
    fprintf('Saved predictions: %s\n', pred_path);
    fprintf('Test loss: %.6f\n', test_loss);

    % Record summary
    summary_backend{end+1, 1} = backend_name;
    summary_num_rows(end+1, 1) = height(data);
    summary_num_train(end+1, 1) = height(train_df);
    summary_num_test(end+1, 1) = height(test_df);
    summary_num_classes(end+1, 1) = nmcl;
    summary_loss(end+1, 1) = test_loss;
    summary_model_path{end+1, 1} = model_path;
end

summary_tbl = table( ...
    summary_backend, ...
    summary_num_rows, ...
    summary_num_train, ...
    summary_num_test, ...
    summary_num_classes, ...
    summary_loss, ...
    summary_model_path, ...
    'VariableNames', { ...
        'Backend', ...
        'NumRows', ...
        'NumTrain', ...
        'NumTest', ...
        'NumClasses', ...
        'TestLoss', ...
        'ModelPath'});

summary_path = fullfile(modelDir, 'qraft_edt_pretrain_summary.csv');
writetable(summary_tbl, summary_path);

fprintf('\nAll backend EDT pretraining finished.\n');
fprintf('Summary saved to: %s\n', summary_path);