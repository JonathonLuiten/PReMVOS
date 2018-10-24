import DeepFried2 as df


def expand_dims(a, axis):
    idx = [slice(None)]*(a.ndim+1)
    idx[axis] = None
    return a[tuple(idx)]


class TripletCriterion(df.Criterion):
    def __init__(self, margin):
        df.Criterion.__init__(self)
        self.margin = margin

    def symb_forward(self, symb_batch, symb_limits):
        anchors = symb_batch[:symb_limits[0]]
        positives = symb_batch[symb_limits[0]:symb_limits[1]]
        negatives = symb_batch[symb_limits[1]:]

        pos_dists = df.T.sqrt(df.T.sum((anchors - positives)**2, axis=1) + 1e-5)
        neg_dists = df.T.sqrt(df.T.sum((anchors - negatives)**2, axis=1) + 1e-5)

        if self.margin is not None:
            return df.T.maximum(pos_dists - neg_dists + self.margin, 0.0)
        else:
            return df.T.nnet.nnet.softplus(pos_dists - neg_dists)


class OutputNormPenalty:
    def __init__(self, module, value=1, axis=-1, eps=1e-8):
        self._mod = module
        self._v = value
        self._axis = axis
        self._eps = eps

    def symb_forward(self):
        x = self._mod._last_symb_out[self._mod._mode]
        norms = df.T.sqrt(self._eps + df.T.sqr(x).sum(axis=self._axis))
        return df.T.mean(abs(norms - self._v))


def cdist_theano(x, eps=1e-8, squared=False):
    # Extend x as x[:,None,...] (but ... doesn't exist in Theano yet.)
    # Then, thanks to broadcasting, we get (B,1,...) - (B,...) = (B,B,...)
    diff = expand_dims(x, 1) - x
    # Finally, sum over all axes but the two first ones.
    #return df.T.sum(diff*diff, axis=tuple(range(2, diff.ndim)))
    dsq = df.T.sum(diff*diff, axis=tuple(range(2, diff.ndim)))
    if squared:
        return dsq
    else:
        return df.T.sqrt(eps + dsq)


class CDist(df.Module):
    def __init__(self, eps=1e-8):
        df.Module.__init__(self)
        self.eps = eps

    def symb_forward(self, x):
        return cdist_theano(x, self.eps)


def test_cdist():
    from scipy.spatial.distance import cdist

    x2 = np.array([[0,1],[2,3],[4,5]], df.floatX)
    np.testing.assert_allclose(CDist(eps=0).forward(x2), cdist(x2,x2))

    x3 = np.random.randn(3,5,10).astype(df.floatX)
    np.testing.assert_allclose(CDist(eps=0).forward(x3), cdist(x3.reshape(3,50), x3.reshape(3,50)))


class OutputDistancePenalty:
    """
    Penalty between 1.0 for identical points and ~0.37 for distance 1 and
    ~0.018 for distance 4.

    Increasing alpha will decrease distance needed to reach penalty, e.g.
    with alpha=4, penalty is ~0.018 for distance 1
    """
    def __init__(self, module, alpha=1):
        self._mod = module
        self._alpha = alpha

    def symb_forward(self):
        symb_x = self._mod._last_symb_out[self._mod._mode]
        dists = cdist_theano(symb_x)

        return df.T.mean(df.T.exp(-self._alpha*dists))


class ContrastiveAllPairsCriterion(df.Criterion):
    """ Forget it, it's not really contrastive either, and it sucks."""
    def __init__(self, avg_nonzeros=False, margin=0.1):
        df.Criterion.__init__(self)
        if avg_nonzeros:
            self.enable_nonzero_averaging()

        self.margin = margin

    def symb_forward(self, symb_x, symb_pids):
        # Flatten all features, so we got (B,F) and everything is easier.
        symb_x = df.T.flatten(symb_x, 2)
        dists = cdist_theano(symb_x)

        # Mask of (B,B) with positive/negative pairs.
        poss = df.T.cast(df.T.eq(symb_pids[:,None], symb_pids), df.floatX)
        pos2 = poss - df.T.eye(poss.shape[0])
        negs = 1.0 - poss

        def per_sample(dists_, poss_, negs_):
            # dists_ is a 1D array of distances
            # poss_ is a 1D array of 1.0 for positives and 0.0 else
            # negs_ is a 1D array of 1.0 for negatives and 0.0 else
            # notice how "self" is not 1.0 in either!
            pos_dists = dists_[poss_.nonzero()]
            neg_dists = dists_[negs_.nonzero()]

            diff = df.T.mean(pos_dists) - df.T.mean(neg_dists)
            if self.margin is not None:
                diff = df.T.maximum(diff - self.margin, 0.0)
            else:
                diff = df.T.nnet.nnet.softplus(diff)
            return diff

        per_sample_loss, _ = df.th.scan(per_sample, sequences=[dists, pos2, negs])
        return per_sample_loss


class AllPairsCriterion(df.Criterion):
    def __init__(self, margin=None, avg_nonzeros=False, squared=False, debug=False):
        df.Criterion.__init__(self)
        if avg_nonzeros:
            self.enable_nonzero_averaging()

        self.margin = margin
        self.squared = squared
        self.debug = debug

    def symb_forward(self, symb_x, symb_pids):
        # Flatten all features, so we got (B,F) and everything is easier.
        symb_x = df.T.flatten(symb_x, 2)

        dists = cdist_theano(symb_x, squared=self.squared)

        # Mask of (B,B) with positive/negative pairs.
        poss = df.T.cast(df.T.eq(symb_pids[:,None], symb_pids), df.floatX)
        negs = df.T.cast(df.T.neq(symb_pids[:,None], symb_pids), df.floatX)
        # negs = 1.0 - poss

        #pos_dists = dists[poss.nonzero()]  # Unfortunately, this flattens.
        #return dists[negs.nonzero()]

        # Find the worst offenders, in a soft way.
        #furthest_pos = df.th.scan(lambda ds, mask: df.T.fn(ds[mask.nonzero()]),
        #                          sequences=[dists, poss])
        #furthest_pos = self.posfn(dists*poss)
        #furthest_pos = df.T.max(dists*poss, axis=1)
        furthest_pos, _ = df.th.scan(lambda ds, mask: df.T.max(ds[mask.nonzero()]), sequences=[dists, poss])
        #closest_neg = self.negfn(dists*negs)
        closest_neg, _ = df.th.scan(lambda ds, mask: df.T.min(ds[mask.nonzero()]), sequences=[dists, negs])
        #closest_neg, _ = df.th.scan(lambda i, ds, mask: df.T.min(ds[i][mask[i].nonzero()]),
        #                            sequences=df.T.arange(dists.shape[0]), non_sequences=[dists, negs])
        if not self.debug:
            diff = furthest_pos - closest_neg
        else:
            diff = furthest_pos / closest_neg

        # For the min/max modes, we only have a number left per sample.
        # For the soft versions, we're still left with a vector per sample,
        # which we'll just average over as a summary. (avg makes batch-indep.)
        #if diff.ndim == 1:
        #    sample_scores = diff
        #else:
        #    sample_scores = df.T.mean(diff, axis=1)

        if self.margin is not None:
            if not self.debug:
                diff = df.T.maximum(diff + self.margin, 0.0)
            else:
                #diff = df.T.maximum(diff, -self.margin)
                #diff = df.T.minimum(diff, 2.0)
                diff = df.T.maximum(diff, 0.5)
        else:
            diff = df.T.nnet.nnet.softplus(diff)

        return diff


class RealAllPairsCriterion(df.Criterion):
    def __init__(self, margin=None, avg_nonzeros=False, topk=False):
        df.Criterion.__init__(self)
        if avg_nonzeros:
            self.enable_nonzero_averaging()

        self.margin = margin
        self.topk = topk

    def symb_forward(self, symb_x, symb_pids):
        # Flatten all features, so we got (B,F) and everything is easier.
        symb_x = df.T.flatten(symb_x, 2)

        dists = cdist_theano(symb_x)

        # Mask of (B,B) with positive/negative pairs.
        poss = df.T.cast(df.T.eq(symb_pids[:,None], symb_pids), dtype='int32')
        pos2 = poss - df.T.eye(poss.shape[0], dtype='int32')
        negs = 1 - poss

        def per_sample(dists_, poss_, negs_):
            # dists_ is a 1D array of distances
            # poss_ is a 1D array of 1.0 for positives and 0.0 else
            # negs_ is a 1D array of 1.0 for negatives and 0.0 else
            # notice how "self" is not 1.0 in either!
            pos_dists = dists_[poss_.nonzero()]
            neg_dists = dists_[negs_.nonzero()]

            # Now compute a (P,N)-matrix of all-to-all (pos - neg) differences.
            all_diff = pos_dists[:,None] - neg_dists[None,:]
            if self.margin is not None:
                all_diff = df.T.maximum(all_diff + self.margin, 0.0)
            else:
                all_diff = df.T.nnet.nnet.softplus(all_diff)

            if self.topk is not False:
                # NOTE: This part only works for fixed-size batches.
                #all_diff = df.T.sort(all_diff, axis=None)[-self.topk:]
                return all_diff

            if self._nonzero_averaging:
                nnz = df.th.gradient.disconnected_grad(all_diff.nonzero_values().shape[0])
                return df.T.sum(all_diff)/(1e-8 + nnz)
            else:
                return df.T.mean(all_diff)

        # Just loop over each sample, because each sample may have a different
        # number of positives/negatives, we can't do this in a tensor-expr.
        per_sample_loss, _ = df.th.scan(per_sample, sequences=[dists, pos2, negs])

        if self.topk is not False:
            per_sample_loss = df.T.sort(per_sample_loss, axis=None)[-self.topk:]

        return per_sample_loss


class LiftedEmbeddingCriterion(df.Criterion):
    def __init__(self, margin):
        df.Criterion.__init__(self)
        self.margin = margin

        # Sanity check
        assert self.margin is not None, "Not thought about this yet brosef."

    def symb_forward(self, symb_x, symb_pids):
        # Flatten all features, so we got (B,F) and everything is easier.
        symb_x = df.T.flatten(symb_x, 2)

        dists = cdist_theano(symb_x)

        # Mask of (B,B) with positive/negative pairs.
        poss = df.T.cast(df.T.eq(symb_pids[:,None], symb_pids), df.floatX)
        pos2 = poss - df.T.eye(poss.shape[0])
        negs = 1.0 - poss

        def per_sample(dists_, poss_, negs_):
            # dists_ is a 1D array of distances
            # poss_ is a 1D array of 1.0 for positives and 0.0 else
            # negs_ is a 1D array of 1.0 for negatives and 0.0 else
            # notice how "self" is not 1.0 in either!
            pos_dists = dists_[poss_.nonzero()]
            neg_dists = dists_[negs_.nonzero()]

            negsum = df.T.log(df.T.sum(df.T.exp(self.margin - neg_dists)))
            possum = df.T.log(df.T.sum(df.T.exp(pos_dists)))

            # That didn't fly so well.
            #if self._nonzero_averaging:
            #    negsum = negsum / df.th.gradient.disconnected_grad(neg_dists.shape[0])
            #    possum = possum / df.th.gradient.disconnected_grad(pos_dists.shape[0])
            j = df.T.maximum(negsum + possum, 0.0)
            return 0.5*j*j

        # Just loop over each sample, because each sample may have a different
        # number of positives/negatives, we can't do this in a tensor-expr.
        per_sample_loss, _ = df.th.scan(per_sample, sequences=[dists, pos2, negs])
        return per_sample_loss


class Xent(df.Criterion):
    def __init__(self, axis, alpha_tgt=0, alpha_out=0, clip=None):
        df.Criterion.__init__(self)
        self.clip = clip
        self.axis = axis
        self.at = alpha_tgt
        self.ao = alpha_out

    def symb_forward(self, symb_out, symb_tgt):
        # TODO: If this is not the case, we could to the 1-hot encoding here,
        #       and even the target-smoothing.
        self._assert_same_dim(symb_out, symb_tgt)

        if self.at > 0:
            symb_tgt = (1-self.at)*symb_tgt + self.at*symb_out
        if self.ao > 0:
            symb_out = (1-self.ao)*symb_out + self.ao*symb_tgt

        if self.clip is not None:
            symb_out = df.T.clip(symb_out, self.clip, 1-self.clip)

        bce = symb_tgt * df.T.log(symb_out)
        bce = df.T.sum(bce, self.axis)
        return -bce


class FisherCriterion(df.Criterion):
    def __init__(self, margin_add=None, margin_mul=None, pull_margin=None, eps=1e-8, negweight=1.0):
        df.Criterion.__init__(self)
        self.madd = margin_add
        self.mmul = margin_mul
        self.pm = pull_margin
        self.eps = eps
        self.negweight = negweight

        assert self.mmul is None, "Currently unsupported combo"

    def symb_forward(self, symb_x, symb_batch_shape):
        # Say we have K "classes/clusters/chunks" with P points each.
        # This reshape both gives us an axis for the clusters and the points
        # as well as flattens all F features, so we get (K,P,F) and everything is easier.
        symb_x = df.T.reshape(symb_x, (symb_batch_shape[0], symb_batch_shape[1], -1))

        # Compute the cluster-means, resulting in (K,1,F)
        symb_means = df.T.mean(symb_x, axis=1, keepdims=True)
        symb_stds = df.T.std(symb_x, axis=1, keepdims=True)

        # Compute the distances to the means, which we want to minimize.
        # Results in a (K,P)
        posd2 = df.T.sum((symb_means - symb_x)**2, axis=-1)
        if self.pm is not None:
            posd2 = df.T.maximum(posd2, self.pm**2)
        else:
            posd2 = self.eps + posd2
        posterms = df.T.sqrt(posd2)
        #return posterms

        # Compute all distances between all means. Gives (K,K)
        meandists = cdist_theano(symb_means[:,0,:], self.eps)
        #return meandists

        # Margin
        if self.madd is not None or self.mmul is not None:
            madd = self.madd or 0.0
            mmul = self.mmul or 1.0

            #if self.msl and self.msr:
            #    mstds = mmul*symb_stds + madd
            #    mstds = mstds + mstds[:,0,:]
            #    # Is now (K,F)
            #    # TODO: what next? Need to undo a step of cdist above!
            #meandists = df.T.maximum(meandists - self.madd, 0.0)
            meandists = df.T.maximum(self.madd - meandists, 0.0)

            # And erase the diagonal!
            meandists *= (1-df.T.eye(meandists.shape[0]))
        #return meandists

        # Take half on the meandists side because we got each one twice.
        return df.T.sum(posterms, axis=1) + df.T.sum(meandists/2, axis=1)*self.negweight


class KLSingleGaussiansMaxCrit(df.Criterion):
    def __init__(self, push_margin=0.1, pull_margin=0.1, invert=True, eps=1e-8):
        df.Criterion.__init__(self)
        self.dm = push_margin**2
        self.pm = pull_margin
        self.inv = invert
        self.eps = eps

    def symb_forward(self, symb_x, symb_batch_shape):
        # Say we have K "classes/clusters/chunks" with P points each.
        # This reshape both gives us an axis for the clusters and the points
        # as well as flattens all F features, so we get (K,P,F) and everything is easier.
        K, P = symb_batch_shape[0], symb_batch_shape[1]
        symb_x = df.T.reshape(symb_x, (K, P, -1))

        # Compute the cluster-means, resulting in (K,1,F)
        means = df.T.mean(symb_x, axis=1, keepdims=True)
        vars_ = df.T.var(symb_x, axis=1, keepdims=True)

        mean_dists = means - means[:,0,:]  # -> (K,K,F)
        mean_dists = mean_dists*mean_dists

        # Put a margin on the mean distances AND on the variance
        if self.pm is not None:
            vars_ = df.T.maximum(vars_, self.pm**2)
        else:
            vars_ = vars_ + self.eps

        t1 = vars_ / vars_[:,0,:]  # -> (K,K,F)
        t2 = mean_dists / vars_[None,:,0,:]  # -> (K,K,F)
        mean_dists = df.T.minimum(t2, 2.0 + self.dm)

        # Ignore the third term with ln as it cancels out due to symmetry
        # Ignore the D constant as it doesn't change anything in the optimization since this is the only loss.

        costs = df.T.sum(t1, axis=-1) + df.T.sum(t2, axis=-1)
        if self.inv:
            costs = 1./(1. + costs)
        else:
            costs = -costs

        # Mask out the diagonals.
        return costs * (1 - df.T.eye(K))


class ZScore(df.Criterion):
    # See http://wwwf.imperial.ac.uk/~naheard/C245/hypothesis_testing_article.pdf
    # 3.2, z = (x-y)/sqrt(v1/n1 + v2/n2)
    #
    # But we compute z^2

    def __init__(self, outer_margin=4.0, pull_margin=0.1, inner_margin=None):
        df.Criterion.__init__(self)
        self.om = outer_margin
        self.pm = pull_margin
        self.im = inner_margin

    def symb_forward(self, symb_x, symb_batch_shape):
        # Say we have K "classes/clusters/chunks" with P points each.
        # This reshape both gives us an axis for the clusters and the points
        # as well as flattens all F features, so we get (K,P,F) and everything is easier.
        K, P = symb_batch_shape[0], symb_batch_shape[1]
        symb_x = df.T.reshape(symb_x, (K, P, -1))

        # Compute the cluster-means, resulting in (K,1,F)
        means = df.T.mean(symb_x, axis=1, keepdims=True)
        vars_ = df.T.var(symb_x, axis=1, keepdims=True)

        mean_dists = means - means[:,0,:]  # -> (K,K,F)
        mean_dists2 = mean_dists*mean_dists

        if self.im is not None:
            mean_dists2 = df.T.minimum(mean_dists2, self.im**2)

        if self.pm is not None:
            vars_ = df.T.maximum(vars_, self.pm**2)

        avg_var = 0.5*(vars_ + vars_[:,0,:])  # -> (K,K,F)
        z2 = mean_dists2/avg_var

        # At this point, z2 is dimension-less and the axes are scaled such that
        # the average of the pairs of clusters have std/var 1
        if self.om is not None:
            z2 = df.T.minimum(z2, self.om**2)

        z2 = df.T.sum(z2, axis=-1)  # -> (K,K)

        costs = 1/(1+z2)
        #costs = -z2

        # Mask out the diagonals.
        return costs * (1 - df.T.eye(K))
