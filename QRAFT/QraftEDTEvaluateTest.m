clear;
clc;

testDataRoot   = './data/qraft_test_data';
modelRoot      = './model/qraft_finetuned_models_edt';
outputRoot     = './qraft_test_results_edt';

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

SEEDS = [1, 2, 3];

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
    'StateUpDnErr75'};

summary_backend = {};
summary_family = {};
summary_seed = [];
summary_num_circuits = [];
summary_mean_hl = [];
summary_result_path = {};

for s = 1:length(SEEDS)
    seed = SEEDS(s);
    fprintf('\n====================================================\n');
    fprintf('Evaluating test data for seed = %d\n', seed);
    fprintf('====================================================\n');

    seedModelDir = fullfile(modelRoot, ['seed_' num2str(seed)]);
    seedOutDir   = fullfile(outputRoot, ['seed_' num2str(seed)]);

    if ~exist(seedOutDir, 'dir')
        mkdir(seedOutDir);
    end

    for b = 1:length(backends)
        backend = backends{b};
        fprintf('\n========================================\n');
        fprintf('Seed: %d | Backend: %s\n', seed, backend);
        fprintf('========================================\n');

        backendTestDir = fullfile(testDataRoot, backend);
        if ~exist(backendTestDir, 'dir')
            fprintf('[WARN] Missing test data dir: %s\n', backendTestDir);
            continue;
        end

        testFiles = dir(fullfile(backendTestDir, [backend '_*_qraft_full.csv']));
        if isempty(testFiles)
            fprintf('[WARN] No test files found in: %s\n', backendTestDir);
            continue;
        end

        for tf = 1:length(testFiles)
            testName = testFiles(tf).name;
            testPath = fullfile(testFiles(tf).folder, testName);

            family = regexprep(testName, ['^' backend '_'], '');
            family = regexprep(family, '_qraft_full\.csv$', '');

            modelPath = fullfile(seedModelDir, sprintf('%s__%s__qraft_edt_finetune.mat', backend, family));
            if ~isfile(modelPath)
                fprintf('[WARN] Missing finetuned model: %s\n', modelPath);
                continue;
            end

            fprintf('\n--- Seed: %d | Backend: %s | Family: %s ---\n', seed, backend, family);
            fprintf('Reading test file: %s\n', testPath);

            T = readtable(testPath);
            T = rmmissing(T);

            requiredCols = [featureCols, {'CircuitID', 'State', 'StateRealProb'}];
            missingCols = setdiff(requiredCols, T.Properties.VariableNames);
            if ~isempty(missingCols)
                error('Missing columns in %s: %s', testPath, strjoin(missingCols, ', '));
            end

            load(modelPath, 'model');

            % Predict row-wise
            X = T(:, featureCols);
            pred = predict(model, X);
            pred = round(pred);
            pred(pred < 0) = 0;
            pred(pred > 100) = 100;

            T.StatePredProb = pred;

            circuitIDs = unique(T.CircuitID);
            circuit_result_rows = [];
            hl_values = [];

            for cid_idx = 1:length(circuitIDs)
                cid = circuitIDs{cid_idx};
                idx = strcmp(T.CircuitID, cid);
                sub = T(idx, :);

                % sort by state so predicted and ideal vectors align
                sub = sortrows(sub, 'State');

                ideal = double(sub.StateRealProb) ./ 100.0;
                predv = double(sub.StatePredProb) ./ 100.0;

                % renormalize both distributions
                if sum(ideal) > 0
                    ideal = ideal ./ sum(ideal);
                end
                if sum(predv) > 0
                    predv = predv ./ sum(predv);
                end

                hl = HellingerDistance(ideal, predv);
                hl_values(end+1, 1) = hl; %#ok<AGROW>

                circuit_result_rows = [circuit_result_rows; ...
                    table({backend}, {family}, seed, {cid}, hl, ...
                    'VariableNames', {'Backend', 'Family', 'Seed', 'CircuitID', 'Hellinger'})]; %#ok<AGROW>
            end

            outPath = fullfile(seedOutDir, sprintf('%s__%s__qraft_test_results.csv', backend, family));
            writetable(circuit_result_rows, outPath);

            mean_hl = mean(hl_values);

            fprintf('Saved results: %s\n', outPath);
            fprintf('Mean Hellinger: %.6f | NumCircuits: %d\n', mean_hl, length(circuitIDs));

            summary_backend{end+1,1} = backend;
            summary_family{end+1,1} = family;
            summary_seed(end+1,1) = seed;
            summary_num_circuits(end+1,1) = length(circuitIDs);
            summary_mean_hl(end+1,1) = mean_hl;
            summary_result_path{end+1,1} = outPath;
        end
    end
end

summary_tbl = table( ...
    summary_backend, ...
    summary_family, ...
    summary_seed, ...
    summary_num_circuits, ...
    summary_mean_hl, ...
    summary_result_path, ...
    'VariableNames', { ...
        'Backend', ...
        'Family', ...
        'Seed', ...
        'NumCircuits', ...
        'MeanHellinger', ...
        'ResultPath'});

summary_path = fullfile(outputRoot, 'qraft_test_eval_summary.csv');
writetable(summary_tbl, summary_path);

fprintf('\nAll Qraft test evaluation finished.\n');
fprintf('Summary saved to: %s\n', summary_path);


function result = HellingerDistance(p, q)
    n = length(p);
    sum_ = 0.0;
    for i = 1:n
        sum_ = sum_ + (sqrt(p(i)) - sqrt(q(i)))^2;
    end
    result = (1.0 / sqrt(2.0)) * sqrt(sum_);
end