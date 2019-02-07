function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

anomalies = pval < epsilon;
[id falseneg] = find(anomalies-yval==-1); %find false negatives
fn = sum(falseneg);
[id falsepos] = find(anomalies-yval==1); %find false positives
fp = sum(falsepos);
[id truepos] = find(anomalies+yval==2); %find true positives (NB where it's +ve
                                        %and we got it right, 1 1)
tp = sum(truepos);
prec = tp/(tp+fp);
rec = tp/(tp+fn);
F1 = (2 * prec * rec)/(prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
