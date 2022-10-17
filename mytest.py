import torch
import torch.nn as nn
import math

def validate_loss(output, target):
    val = 0
    for li_x, li_y in zip(output, target):
        for i, xy in enumerate(zip(li_x, li_y)):
            x, y = xy
            loss_val = y * (math.log(y, math.e) - x)
            val += loss_val
    return val / output.nelement()

torch.manual_seed(20)
loss = nn.KLDivLoss()
input = torch.Tensor([[1, 1, 0.], [1, 1, 0.], [0., 1., 1.], [1, 1, 1.]]).log()
target = torch.Tensor([[1, 1, 0.], [1, 1, 0.], [0., 1., 1.], [1, 1, 1.]])
output = loss(input, target)
print("default loss:", output)

# output = validate_loss(input, target)
# print("validate loss:", output)

loss = nn.KLDivLoss(reduction="batchmean")
output = loss(input, target)
print("batchmean loss:", output)

loss = nn.KLDivLoss(reduction="mean")
output = loss(input, target)
print("mean loss:", output)

loss = nn.KLDivLoss(reduction="none")
output = loss(input, target)
print("none loss:", output)
