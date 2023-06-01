import numpy as np


def length_norm(mat):
    return mat / np.sqrt(np.sum(mat * mat, axis=1))[:, None]


def compute_pmiss_pfa_rbst(scores, labels, weights=None):
    """ computes false positive rate (FPR) and false negative rate (FNR)
    given trial socres and their labels. A weights option is also provided
    to equalize the counts over score partitions (if there is such
    partitioning).
    """

    sorted_ndx = np.argsort(scores)
    labels = labels[sorted_ndx]
    if weights is not None:
        weights = weights[sorted_ndx]
    else:
        weights = np.ones((labels.shape), dtype='f8')

    tgt_wghts = weights * (labels == 1).astype('f8')
    imp_wghts = weights * (labels == 0).astype('f8')

    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)
    return fnr, fpr


def compute_eer(fnr, fpr):
    """ computes the equal error rate (EER) given FNR and FPR values calculated
        for a range of operating points on the DET curve
    """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
    return fnr[x1] + a * (fnr[x2] - fnr[x1])


def compute_c_norm(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ computes normalized minimum detection cost function (DCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    """

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det, c_det_ind = min(dcf), np.argmin(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det/c_def, c_det_ind


def compute_equalized_min_cost(labels, scores, ptar=[0.01, 0.001]):
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer = compute_eer(fnr, fpr)
    min_c = 0.
    for pt in ptar:
        tmp, idx = compute_c_norm(fnr, fpr, pt)
        min_c += tmp
    return eer*100, min_c / len(ptar)
    
class SVevaluation(object):
    def __init__(self, enrol_utt, test_utt, trial_file, ptar=[0.01, 0.001]):
        # trials file format: is_target(0 or 1) enrol_utt test_utt
        self.ptar = ptar
        self.labels = [int(line.split()[0]) for line in open(trial_file)]
        
        enrol_utt_idx = {utt:i for i, utt in enumerate(enrol_utt)}
        test_utt_idx = {utt:i for i, utt in enumerate(test_utt)}
        self.trial_idx = [[enrol_utt_idx.get(line.split()[1]), test_utt_idx.get(line.split()[2])] for line in open(trial_file)]
        bad_idx = [i for i, ti in enumerate(self.trial_idx) if None in ti]
        for i in sorted(bad_idx, reverse=True):
            del self.trial_idx[i], self.labels[i]
        self.labels = np.array(self.labels)
    
    def eer_cost(self, enrol_embd, test_embd=None, scoring='cosine', score_norm=None, cohort=None, num_cohort=None):
        enrol_embd = length_norm(enrol_embd)
        test_embd = enrol_embd if test_embd is None else length_norm(test_embd)
        
        if score_norm:
            cohort = length_norm(cohort)
            score_enrol_cohort = enrol_embd @ cohort.T
            enrol_idx = score_enrol_cohort.argsort()[:, ::-1]
            score_test_cohort = test_embd @ cohort.T
            test_idx = score_test_cohort.argsort()[:, ::-1]
            if score_norm == 'asnorm1':
                scores = [self.score_norm((enrol_embd[i] * test_embd[j]).sum(),
                                          score_enrol_cohort[i, enrol_idx[i, :num_cohort]],
                                          score_test_cohort[j, test_idx[j, :num_cohort]])
                          for i, j in self.trial_idx]
            elif score_norm == 'asnorm2':
                scores = [self.score_norm((enrol_embd[i] * test_embd[j]).sum(),
                                          score_enrol_cohort[i, test_idx[j, :num_cohort]],
                                          score_test_cohort[j, enrol_idx[i, :num_cohort]])
                          for i, j in self.trial_idx]
        else:
            scores = [(enrol_embd[i] * test_embd[j]).sum() for i, j in self.trial_idx]
            
        eer, cost = compute_equalized_min_cost(self.labels, np.array(scores), self.ptar)
        return eer, cost
    
    def score_norm(self, score, score_enrol_cohort, score_test_cohort):
        return (score - score_enrol_cohort.mean()) / score_enrol_cohort.std() + \
               (score - score_test_cohort.mean()) / score_test_cohort.std()
        