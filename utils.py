import torch
import numpy as np
from settings import parse_args

args = parse_args()

class Cutout(object):
    '''Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    '''
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        '''
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        '''
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class NClassRandomSampler(torch.utils.data.sampler.Sampler):
    r'''Samples elements such that most batches have N classes per batch.
    Elements are shuffled before each epoch.

    Arguments:
        targets: target class for each example in the dataset
        n_classes_per_batch: the number of classes we want to have per batch
    '''
    def __init__(self, targets, n_classes_per_batch, batch_size):
        self.targets = targets
        self.n_classes = int(np.max(targets))
        self.n_classes_per_batch = n_classes_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        n = self.n_classes_per_batch
        
        ts = list(self.targets)
        ts_i = list(range(len(self.targets)))
        
        np.random.shuffle(ts_i)
        #algorithm outline: 
        #1) put n examples in batch
        #2) fill rest of batch with examples whose class is already in the batch
        while len(ts_i) > 0:
            idxs, ts_i = ts_i[:n], ts_i[n:] #pop n off the list
                
            t_slice_set = set([ts[i] for i in idxs])
            
            #fill up idxs until we have n different classes in it. this should be quick.
            k = 0
            while len(t_slice_set) < 10 and k < n*10 and k < len(ts_i):
                if ts[ts_i[k]] not in t_slice_set:
                    idxs.append(ts_i.pop(k))
                    t_slice_set = set([ts[i] for i in idxs])
                else:
                    k += 1
            
            #fill up idxs with indexes whose classes are in t_slice_set.
            j = 0
            while j < len(ts_i) and len(idxs) < self.batch_size:
                if ts[ts_i[j]] in t_slice_set:
                    idxs.append(ts_i.pop(j)) #pop is O(n), can we do better?
                else:
                    j += 1
            
            if len(idxs) < self.batch_size:
                needed = self.batch_size-len(idxs)
                idxs += ts_i[:needed]
                ts_i = ts_i[needed:]
                    
            for i in idxs:
                yield i

    def __len__(self):
        return len(self.targets)
    
def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not args.no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R

