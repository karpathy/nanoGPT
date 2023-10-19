import torch

# def get_interval_values(X, intervals):
#     # Create a mask that represents the interval for each row, inclusive both ends
#     mask = torch.arange(X.size(1)).unsqueeze(0).expand((X.size(0), X.size(1))).to(intervals.device)
#     mask = (mask >= intervals[:, 0].unsqueeze(1)) & (mask <= intervals[:,1].unsqueeze(1))
#     assert X.dim() == 3 or X.dim() == 2
#     if X.dim() == 3:
#         mask = mask.unsqueeze(2)
#         ret = torch.masked_select(X, mask).reshape(-1, X.shape[2])
#     else:
#         ret = torch.masked_select(X, mask)
#     return ret

def get_interval_values(X, intervals):
    frags = []
    for i in range(len(intervals)):
        frags.append(X[i, intervals[i, 0]: intervals[i, 1]])
    return torch.cat(frags, dim=0)