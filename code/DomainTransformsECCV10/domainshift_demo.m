% Demo script for domain adaptation using learned transforms
% Authors: K. Saenko and B. Kulis

% Edit the following to set location of dataset
datadir = '/u/vis/x1/office/';

% Select a configuration file, e.g. :
addpath ./config_files
%config_file = 'config_samecat_webcam_dslr';
%config_file = 'config_samecat_dslr_webcam';
config_file = 'config_diffcat_webcam_dslr';
%config_file = 'config_diffcat_dslr_webcam';

%%%%%%%%%%%%%%%%%%%%%%%% load config %%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('config_file')
    disp('Please set the config_file variable in this script before running');
    return;
end

PARAM = eval([config_file '(datadir)'])

%%%%%%%%%%%%%%%%%%%%%%%% load data %%%%%%%%%%%%%%%%%%%%%%%%

if ~isdir(datadir)
    disp('Please set the dataset directory in this script before running');
    return;
end

[Data Labels Files Objs Matches] = loadDomains(PARAM);
PARAM.Matches = Matches;

% set domains A and B
XA = Data{1};
yA = Labels{1};

XB = Data{2};
yB = Labels{2};

%number of NNs to use for the kNN classifier
k = PARAM.k;

%gamma values to try (weight parameter)
gamma_set = PARAM.gamma_set;

[nA,dA] = size(XA);
[nB,dB] = size(XB);

%%%%%%%%%%%%%%%%%%% main experiment, run several times %%%%%%%%%%%%%%%%%%%%

NUM_RUNS = PARAM.NUM_RUNS;

for j = 1:NUM_RUNS
    disp(sprintf('Run %d',j));

    if PARAM.testOnNewCategories
        % trexs and testexs have different classes
        [trexsA, testexsA, trexsB, testexsB]= splitDiffCategories(yA, yB, Objs, PARAM);
        trknnA = testexsA;  % train knn on novel classes only
    else
        [trexsA, testexsA, trexsB, testexsB]= splitAllCategories(yA, yB, Objs, PARAM);
        trknnA = trexsA;  % train knn on the same data as the transform
    end
    
    % grab dslr/webcam instance matches for training set (for correspondence constraints)
    MatchesTrain = [];
    for kk=1:size(Matches,1)
        ii = Matches(kk,1);
        jj = Matches(kk,2);
        % add the match to constraints if both pts are in train set
        if ismember(ii,trexsA) && ismember(jj,trexsB)
            MatchesTrain = [MatchesTrain; find(ii==trexsA) find(jj==trexsB)];
        end
    end
    PARAM.MatchesTrain = MatchesTrain;
    
    % subsample all pairs of possible constraints
    PARAM.constraint_num = length(trexsB)^2;
   
    % no transform/metric, knn-classify target using only source
    PARAM.S = [];
    knn.A_B(j) = symmetricKNN(XA(trknnA,:),  yA(trknnA), ...
			      XB(testexsB,:),yB(testexsB),PARAM);

    % no transform/metric, knn-classify target using source+target
    knn.AB_B(j)= symmetricKNN([XA(trknnA,:); XB(trexsB,:)],...
			      [yA(trknnA) yB(trexsB)], ...
			      XB(testexsB,:),yB(testexsB),PARAM);

    % test different transformations          
    for g=1:length(gamma_set)
        PARAM.gamma = gamma_set(g);
        
        % ITML: learn same-domain metric, only on B
        PARAM.constraint_type = 'allpairs';
        PARAM = learnSymmTransform([],[], XB(trexsB,:),  yB(trexsB), PARAM);
        % classify target using source
        knn.m_B_A_B(j,g) = symmetricKNN(XA(trknnA,:),  yA(trknnA), ...
            XB(testexsB,:),yB(testexsB),PARAM);
        % classify target using target (train knn on B)
        knn.m_B_B_B(j,g) = symmetricKNN(XB(trexsB,:),  yB(trexsB), ...
            XB(testexsB,:),yB(testexsB),PARAM);
        
        % ITML: learn same-domain metric, on A+B
        PARAM.constraint_type = 'allpairs';
        PARAM = learnSymmTransform(XA(trexsA,:), yA(trexsA), ...
            XB(trexsB,:), yB(trexsB), PARAM);
        % classify target using source A
        knn.m_AB_A_B(j,g) = symmetricKNN(XA(trknnA,:),  yA(trknnA), ...
            XB(testexsB,:),yB(testexsB),PARAM);
        % classify using A+B (makes sense if trexsB has same categories)
        knn.m_AB_AB_B(j,g) = symmetricKNN([XA(trknnA,:); XB(trexsB,:)],...
            [yA(trknnA) yB(trexsB)], ...
            XB(testexsB,:),yB(testexsB),PARAM);
        
        % Symm: learn cross-domain symmetric transform on A->B
        PARAM.constraint_type = PARAM.symm_constraint_type;
        PARAM = learnSymmTransform(XA(trexsA,:), yA(trexsA), ...
            XB(trexsB,:), yB(trexsB), PARAM);
        % classify target using source only
        knn.s_AB_A_B(j,g) = symmetricKNN(XA(trknnA,:),  yA(trknnA), ...
            XB(testexsB,:),yB(testexsB),PARAM);
        
        % Asymm: learn asymmetric transform on A->B
        PARAM.constraint_type = PARAM.asymm_constraint_type; 
        PARAM = learnAsymmTransform(XA(trexsA,:), yA(trexsA), ...
            XB(trexsB,:), yB(trexsB), PARAM);        
        % classify target using source only
        knn.a_AB_A_B(j,g) = asymmetricKNN(XA(trknnA,:),yA(trknnA), ...
            XB(testexsB,:),yB(testexsB),PARAM);
     end
end

PARAM

result_file = [mfilename '-' config_file '-' ...
    sprintf('%d.%d.%d.%02d-%02d-%02d', fix(clock)) '.mat'];
save(result_file, 'knn', 'PARAM');
fprintf('\nSaving results to %s\n', result_file);

gamma_set = PARAM.gamma_set;
if length(gamma_set)>1
    figure
    hold on;
    plot(log10(gamma_set), mean(knn.AB_B)*ones(size(gamma_set)),'k-.');
    errorbar(log10(gamma_set), mean(knn.m_B_B_B), std(knn.m_B_B_B), 'ko-');
    errorbar(log10(gamma_set), mean(knn.m_AB_A_B), std(knn.m_AB_A_B), 'bo-');
    errorbar(log10(gamma_set), mean(knn.s_AB_A_B), std(knn.s_AB_A_B), 'ro-');
    errorbar(log10(gamma_set), mean(knn.a_AB_A_B), std(knn.a_AB_A_B), 'mo-');
    grid on
    legend('Euclid(A+B)', 'ITML(B)', 'ITML(A+B)', 'Symm(A+B)', 'Asymm(A+B)');
    xlabel(sprintf('kNN (k=%d) trained on domain A (%s), tested on domain B (%s)', k,...
        strrep(PARAM.image_dirs{1},'/images/',''), strrep(PARAM.image_dirs{2},'/images/','')));
    title(strrep(result_file, '_', '\_'));
else
    fprintf('\nAccuracy of kNN (k=%d) on B, for domain A=%s, domain B=%s\n\n', k,...
        strrep(PARAM.image_dirs{1},'/images/',''), strrep(PARAM.image_dirs{2},'/images/',''));
    fprintf('     method:  Euclid    Euclid  ITML    ITML    Asymm   Symm\n');
    fprintf('--------------------------------------------------------------\n');
    fprintf('tform train:  n/a       n/a     A+B     B       A+B     A+B   \n'); 
    fprintf(' train data:  A         A+B     A       B       A       A     \n');
    fprintf('   accuracy:  %.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f \n',...
        mean(knn.A_B),mean(knn.AB_B),mean(knn.m_AB_A_B),mean(knn.m_B_B_B),...
        mean(knn.a_AB_A_B),mean(knn.s_AB_A_B));
end

