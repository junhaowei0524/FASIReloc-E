import torch
# from torch_scatter import scatter_max

def farthest_point_sample(data,npoints):

    N,D = data.shape 
    xyz = data[:,:3] 
    centroids = torch.zeros(size=(npoints,)) 
    dictance = torch.ones(size=(N,))*1e10  
    farthest = torch.ones(size=(1,)) 
    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest,:]
        dict = ((xyz-centroid)**2).sum(dim=-1)
        mask = dict < dictance
        dictance[mask] = dict[mask]
        farthest = torch.argmax(dictance,dim=-1)
    print(centroids.type(torch.long))
    data= data[centroids.type(torch.long)]
    return data



@torch.no_grad()
def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density

@torch.no_grad()
def s_fps(data, score, npoints):

    
    N,D = data.shape  
    xyz = data[:,:3] 
    centroids = torch.zeros(size=(npoints,), device=xyz.device) 
    dictance = torch.ones(size=(N,), device=xyz.device)*1e10
    farthest = torch.ones(size=(1,), device=xyz.device).int()
    if score is None:
        score = torch.ones(size=(N,), device=xyz.device)

    for i in range(npoints):
        centroids[i] = farthest
        centroid = xyz[farthest,:]
        dict = ((xyz-centroid)**2).sum(dim=-1)
        mask = dict < dictance
        dictance[mask] = dict[mask]
        weighted_dictance = dictance * score
        farthest = torch.argmax(weighted_dictance, dim=-1)
    print(centroids.type(torch.long))
    data= data[centroids.type(torch.long)]
    return centroids.type(torch.long)






    