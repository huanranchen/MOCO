from moco import MoCo
from tqdm import tqdm
import torch

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# a = MoCo().to(device)
# a.train()
#
# from transfer import train
#
# train(mode = 'q_encoder')

from matplotlib import pyplot as plt
import numpy as np

x = np.arange(10)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x, x*x)
plt.plot(x, 2*x)

plt.legend(['x^2','2x'])
plt.show()