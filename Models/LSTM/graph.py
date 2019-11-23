import pickle
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


with open('Joint_Encoder_Model_Backup/train_vals', 'rb') as fp:
    train_data = pickle.load(fp)

loss_list = train_data[5]
acc_list = train_data[6]
F1_list = train_data[7]
val_loss_list = train_data[8]
val_acc_list = train_data[9]
val_F1_list = train_data[10]

# %%

y1 = val_acc_list
y2 = acc_list

x = np.arange(1, len(y1)+1, 1)  # (1 = starting epoch, len(y1) = no. of epochs, 1 = step)

plt.plot(x, y1, 'b', label='Validation Accuracy')
plt.plot(x, y2, 'r', label='Training Accuracy')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.show()

# %%

y1 = val_loss_list
y2 = loss_list

plt.plot(x, y1, 'b', label='Validation Loss')
plt.plot(x, y2, 'r', label='Training Loss')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.show()

# %%

y1 = val_F1_list
y2 = F1_list

# print(y2)

plt.plot(x, y1, 'b', label='Validation F1')
#plt.plot(x, y2, 'r', label='Training F1')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.show()
