import numpy as np



### for the dataloading part
class semi_supervised_sampler(object):
    
    def __init__(self,indices_subset_A,indices_subset_B,batch_size,ds_size,p_A = 0.5,fluct=1.0,seed=None):

        self.subset_A = indices_subset_A
        self.subset_B = indices_subset_B
        self.batch_size = batch_size
        self.ds_size = ds_size
        self.p_A = p_A
        self.fluct = fluct
        np.random.seed(seed)
        # NOTE in the current setup this sampler rounds the last batch-up to a full batch
        # in contrast to the dataloader which spits out a fractional batch
        self.n_batches = int(np.ceil(self.ds_size/self.batch_size))
    
    def sample(self):

        indices_sampler = []
        n = 0; n_A = 0; n_B = 0
        
        for _ in range(self.n_batches): 
            indices_batch = []
            # compute the number of samples of subset A in batch
            # if fluct >= 1 this always corresponds in average to the proportion p_A 
            N_A = round(self.batch_size*self.p_A+self.fluct*(np.random.rand()-0.5))

            for i in range(self.batch_size):
                # re-shuffling subsets if every item is drawn
                if n_A == 0: self.subset_A = np.random.permutation(self.subset_A)
                if n_B == 0: self.subset_B = np.random.permutation(self.subset_B) 

                if i < N_A:
                    indices_batch.append(self.subset_A[n_A])
                    n_A += 1
                    n_A = n_A % len(self.subset_A)
                else:
                    indices_batch.append(self.subset_B[n_B])
                    n_B += 1
                    n_B = n_B % len(self.subset_B)
                n += 1

            # shuffle indices within batch and append to sampler
            indices_batch = np.random.permutation(indices_batch)
            indices_sampler = np.append(indices_sampler,indices_batch)

        return indices_sampler.astype(np.int64).tolist()






### for creating datasets
def random_subset(array,p = 1,seed = 0):
    np.random.seed(seed)
    return np.random.choice(array,int(len(array)*p),replace=False)



def compute_distribution(labs_one_hot):

    counts = np.sum(labs_one_hot,axis=0)

    return counts/sum(counts)



def make_uniform(dist,alpha):

    dist_uniform = np.ones(len(dist))/len(dist)

    return alpha*dist_uniform + (1-alpha)*dist



def random_sampling_with_distribution(set_labs,n_samples,class_proportions,print_report=True,seed=None):
    '''
    This function randomly samples a subset from an original set 
    (where each element can be assigned to multiple classes - stored in set_labs).
    The subset should match as close as possible a desired distribution of classes.
    Note: each element can be sampled only once
    '''

    # initialize
    remaining_set_ids = np.arange(len(set_labs))
    n_masks = np.copy(n_samples)
    choosen_ids = []
    choosen_counts = np.zeros(len(class_proportions))
    np.random.seed(seed)

    for i in range(n_samples):

        # compute the new sampling probability per class
        masks_per_class = class_proportions*n_masks             # total desired masks per class
        open_masks_per_class = masks_per_class - choosen_counts # masks per class which still should be sampled
        open_masks_per_class[open_masks_per_class<0] = 0
        sampling_prob_class = open_masks_per_class/sum(open_masks_per_class)

        # compute the sampling probability of each element
        # note: if a element contains multiple classes we assign the sampling prob of the class with the lowest prob
        #       this will prevent an oversampling of that particular class
        #       and prevent that a sample corresponding to a class only consists of images with other classes displayed
        weights = np.multiply(set_labs[remaining_set_ids],sampling_prob_class)
        weights[set_labs[remaining_set_ids] == 0] = np.infty # exclude not visible classes by setting to infinity
        sampling_prob = np.min(weights,axis=1)
        sampling_prob[sampling_prob == np.infty] = 0

        # hierachichal sample starting with class with highest sampling probability
        sampling_prob = sampling_prob*(set_labs[remaining_set_ids,np.argmax(sampling_prob_class)] > 0)
        # print(sampling_prob_class)

        if np.sum(sampling_prob) > 10**-6:
            sampling_prob /= sum(sampling_prob)

            # randomly sample an element
            choosen_id = np.random.choice(remaining_set_ids,1,p=sampling_prob)[0]
            choosen_lab = set_labs[choosen_id]

            # store and remove from id from original set
            choosen_ids.append(choosen_id)
            remaining_set_ids = np.delete(remaining_set_ids,np.where(remaining_set_ids == choosen_id))

            # update sampled statistics
            choosen_counts = choosen_counts + choosen_lab
            n_masks = n_masks + (sum(choosen_lab) - 1)
            # print("n masks =", n_masks)

    if print_report:
        # compare statistics of desired class proportions and actual drawn class proportions
        print('\nSAMPLING WITH DITRIBUTION')
        print('Number of images with label', np.sum(np.sum(set_labs,axis=1) >0))
        print('desired class proportions',np.round(class_proportions,2),'with number of images sampled',n_samples)
        print('actual  class proportions',np.round(choosen_counts/sum(choosen_counts),2),'with number of images sampled',len(choosen_ids))
        print('\n')
        # due to the discrete nature of classes and samples with multiple classes per image 
        # this might not always be exactly possible

    return choosen_ids