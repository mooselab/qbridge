clear;
clc;

pretrainDataDir = './data/qraft_training_data';
tuneDataRoot    = './data/qraft_tuning_data';
outputRoot      = './model/qraft_finetuned_models_edt';

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

SEEDS = [1, 2, 3];
FAMILY_WEIGHT = 3;

backends = { ...
    'FakeAlmaden', 'FakeBoeblingen', 'FakeBrooklyn', 'FakeCairo', ...
    'FakeCambridge', 'FakeCambridgeAlternativeBasis', 'FakeCasablanca', ...
    'FakeGuadalupe', 'FakeHanoi', 'FakeJakarta', 'FakeJohannesburg', ...
    'FakeKolkata', 'FakeLagos', 'FakeManhattan', 'FakeMontreal', ...
    'FakeMumbai', 'FakeNairobi', 'FakeParis', 'FakeRochester', ...
    'FakeSingapore', 'FakeSydney', 'FakeToronto', 'FakeWashington'};

featureCols = { ...
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
    'StateRealProb'};

summary_backend = {};
summary_family = {};
summary_seed = [];
summary_num_rows = [];
summary_num_train = [];
summary_num_test = [];
summary_num_classes = [];
summary_loss = [];
summary_model_path = {};

for s = 1:length(SEEDS)
    seed = SEEDS(s);
    rng(seed);

    fprintf('\n====================================================\n');
    fprintf('Starting finetuning for seed = %d\n', seed);
    fprintf('====================================================\n');

    seedDir = fullfile(outputRoot, ['seed_' num2str(seed)]);
    if ~exist(seedDir, 'dir')
        mkdir(seedDir);
    end

    for b = 1:length(backends)
        backend = backends{b};
        fprintf('\n========================================\n');
        fprintf('Seed: %d | Backend: %s\n', seed, backend);
        fprintf('========================================\n');

        pretrainFile = fullfile(pretrainDataDir, [backend '_qraft_model.csv']);
        if ~isfile(pretrainFile)
            fprintf('[WARN] Missing pretrain data: %s\n', pretrainFile);
            continue;
        end

        pretrainTbl = readtable(pretrainFile);
        pretrainTbl = rmmissing(pretrainTbl);
        pretrainTbl = pretrainTbl(:, featureCols);
        pretrainTbl.StateRealProb = round(pretrainTbl.StateRealProb);
        pretrainTbl.StateRealProb(pretrainTbl.StateRealProb < 0) = 0;
        pretrainTbl.StateRealProb(pretrainTbl.StateRealProb > 100) = 100;

        tuneDir = fullfile(tuneDataRoot, backend);
        if ~exist(tuneDir, 'dir')
            fprintf('[WARN] Missing tuning dir: %s\n', tuneDir);
            continue;
        end

        tuneFiles = dir(fullfile(tuneDir, [backend '_*_qraft_model.csv']));
        if isempty(tuneFiles)
            fprintf('[WARN] No tuning files found in: %s\n', tuneDir);
            continue;
        end

        for tf = 1:length(tuneFiles)
            tuneName = tuneFiles(tf).name;
            tunePath = fullfile(tuneFiles(tf).folder, tuneName);

            family = regexprep(tuneName, ['^' backend '_'], '');
            family = regexprep(family, '_qraft_model\.csv$', '');

            fprintf('\n--- Seed: %d | Backend: %s | Family: %s ---\n', seed, backend, family);
            fprintf('Reading tuning file: %s\n', tunePath);

            tuneTbl = readtable(tunePath);
            tuneTbl = rmmissing(tuneTbl);

            missingCols = setdiff(featureCols, tuneTbl.Properties.VariableNames);
            if ~isempty(missingCols)
                error('Missing columns in %s: %s', tunePath, strjoin(missingCols, ', '));
            end

            tuneTbl = tuneTbl(:, featureCols);
            tuneTbl.StateRealProb = round(tuneTbl.StateRealProb);
            tuneTbl.StateRealProb(tuneTbl.StateRealProb < 0) = 0;
            tuneTbl.StateRealProb(tuneTbl.StateRealProb > 100) = 100;

            combinedTbl = pretrainTbl;
            for rep = 1:FAMILY_WEIGHT
                combinedTbl = [combinedTbl; tuneTbl]; %#ok<AGROW>
            end

            data = combinedTbl(randperm(height(combinedTbl)), :);

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

            mscl = zeros(nmcl);
            for i = 1:nmcl
                for j = 1:nmcl
                    mscl(i, j) = ((clTrn(i) - clTrn(j))^2) * (clTrn(i) + 1);
                end
            end

            % Use a nonstratified CV partition for hyperparameter optimization
            % This avoids warnings when rare classes are missing in some folds.
            cvp = cvpartition(height(train_df), 'KFold', 3);

            model = fitcensemble( ...
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

            test_loss = loss(model, test_df);
            test_df.StatePredProb = predict(model, test_df);

            model_path = fullfile(seedDir, sprintf('%s__%s__qraft_edt_finetune.mat', backend, family));
            pred_path  = fullfile(seedDir, sprintf('%s__%s__qraft_edt_finetune_predictions.csv', backend, family));

            save(model_path, 'model', 'clTrn', 'mscl', 'backend', 'family', 'seed');
            writetable(test_df, pred_path);

            fprintf('Saved model: %s\n', model_path);
            fprintf('Saved predictions: %s\n', pred_path);
            fprintf('Test loss: %.6f\n', test_loss);

            summary_backend{end+1,1} = backend;
            summary_family{end+1,1} = family;
            summary_seed(end+1,1) = seed;
            summary_num_rows(end+1,1) = height(data);
            summary_num_train(end+1,1) = height(train_df);
            summary_num_test(end+1,1) = height(test_df);
            summary_num_classes(end+1,1) = nmcl;
            summary_loss(end+1,1) = test_loss;
            summary_model_path{end+1,1} = model_path;
        end
    end
end

summary_tbl = table( ...
    summary_backend, ...
    summary_family, ...
    summary_seed, ...
    summary_num_rows, ...
    summary_num_train, ...
    summary_num_test, ...
    summary_num_classes, ...
    summary_loss, ...
    summary_model_path, ...
    'VariableNames', { ...
        'Backend', ...
        'Family', ...
        'Seed', ...
        'NumRows', ...
        'NumTrain', ...
        'NumTest', ...
        'NumClasses', ...
        'TestLoss', ...
        'ModelPath'});

summary_path = fullfile(outputRoot, 'qraft_edt_finetune_summary.csv');
writetable(summary_tbl, summary_path);

fprintf('\nAll Qraft EDT family finetuning finished.\n');
fprintf('Summary saved to: %s\n', summary_path);