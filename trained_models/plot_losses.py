from os import truncate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loss = pd.read_csv('trained_models/loss_full.csv')

plt.plot(np.arange(20, 100), loss['loss_train'][20:100], label='Training loss')
plt.plot(np.arange(20, 100), loss['loss_val'][20:100], label='Testing loss')
plt.title('Evolution of GCN losses during training')
plt.xlabel('epoch')
plt.show()