import os
import matplotlib.pyplot as plt 

def plot_learning_curves(loss, val_mae, dir_to_save):
    # plot learning curves
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(loss, label='train loss', color='tab:blue')
    ax1.legend(loc = 'upper right')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(val_mae, label='val mae', color='tab:orange')
    ax2.legend(loc = 'upper right')
    # ax2.set_ylim((0,50))
    fig.savefig(os.path.join(dir_to_save, 'learning_curves.png'), bbox_inches='tight', dpi = 300)
    plt.close()
    