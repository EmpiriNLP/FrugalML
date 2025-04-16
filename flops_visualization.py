import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_vs_flop(flop_b1, flop_b2, steps_b1, steps_b2):
    epoch = np.array([1, 2, 3])
    flop_b1_epoch = np.array([np.sum(flop_b1[0: int(len(steps_b1)/3)]), np.sum(flop_b1[0: int(2*len(steps_b1)/3)]), np.sum(flop_b1[0: int(len(steps_b1))])])
    flop_b2_epoch = np.array([np.sum(flop_b2[0: int(len(steps_b2)/3)]), np.sum(flop_b2[0: int(2*len(steps_b2)/3)]), np.sum(flop_b2[0: int(len(steps_b2))])])

    accuracy_b1 = [27.18, 31.51, 24.36]
    accuracy_b2 = [26.21, 31.77, 28.07]

    plt.scatter(flop_b1_epoch, accuracy_b1, label='Batch Size 1 Accuracy', color='darkorange', marker='o')
    plt.scatter(flop_b2_epoch, accuracy_b2, label='Batch Size 2 Accuracy', color='navy', marker='o')
    plt.xlabel('FLOs')
    plt.ylabel('Accuracy (%)')
    plt.xscale('log')
    plt.title('Accuracy vs FLOs')
    plt.legend()
    plt.grid(axis='y', alpha=0.4)
    plt.show()

    plt.savefig('accuracy_vs_flos.png', dpi=300, bbox_inches='tight')

def plot_resources_vs_flop(flop_b1, flop_b2, cuda_time_b1, cuda_time_b2, cpu_time_b1, cpu_time_b2, memory_b1, memory_b2, steps_b1, steps_b2):
    flop_b1_cum_epoch = flop_b1.cumsum()
    flop_b2_cum_epoch = flop_b2.cumsum()

    cuda_time_b1_cum_epoch = cuda_time_b1.cumsum()
    cuda_time_b2_cum_epoch = cuda_time_b2.cumsum()

    cpu_time_b1_cum_epoch = cpu_time_b1.cumsum()
    cpu_time_b2_cum_epoch = cpu_time_b2.cumsum()

    memory_b1_cum_epoch = memory_b1.cumsum()
    memory_b2_cum_epoch = memory_b2.cumsum()

    epoch_steps = [0, 6200, 12400, 18600]
    epochs = [0, 1, 2, 3]

    fig, axs =  plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    axs[0].plot(steps_b1, flop_b1_cum_epoch, label='Batch Size 1', color='darkorange')
    axs[0].plot(steps_b1, flop_b2_cum_epoch, label='Batch Size 2', color='navy')
    axs[0].set_xticks(epoch_steps, epochs)
    axs[0].set_xlabel('Epoch', fontsize=12)
    axs[0].set_ylabel('FLOs', fontsize=12)
    axs[0].set_title('Floating Point Operations vs Epochs', fontsize=16)
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[0].grid(True, alpha=0.4)

    axs[1].plot(steps_b1, cuda_time_b1_cum_epoch, label='Batch Size 1', color='darkorange')
    axs[1].plot(steps_b1, cuda_time_b2_cum_epoch, label='Batch Size 2', color='navy')
    axs[1].set_xticks(epoch_steps, epochs)
    axs[1].set_xlabel('Epochs', fontsize=12)
    axs[1].set_ylabel('CUDA Time (ms)', fontsize=12)
    axs[1].set_title('CUDA Time vs Epochs', fontsize=16)
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].grid(True, alpha=0.4)

    axs[2].plot(steps_b1, cpu_time_b1_cum_epoch, label='Batch Size 1', color='darkorange')
    axs[2].plot(steps_b1, cpu_time_b2_cum_epoch, label='Batch Size 2', color='navy')
    axs[2].set_xticks(epoch_steps, epochs)
    axs[2].set_xlabel('Epochs', fontsize=12)
    axs[2].set_ylabel('CPU Time (ms)', fontsize=12)
    axs[2].set_title('CPU Time vs Epochs', fontsize=16)
    axs[2].set_yscale('log')
    axs[2].legend()
    axs[2].grid(True, alpha=0.4)

    axs[3].plot(steps_b1, memory_b1_cum_epoch, label='Batch Size 1', color='darkorange')
    axs[3].plot(steps_b1, memory_b2_cum_epoch, label='Batch Size 2', color='navy')
    axs[3].set_xticks(epoch_steps, epochs)
    axs[3].set_xlabel('Epochs', fontsize=12)
    axs[3].set_ylabel('Memory (MB)', fontsize=12)
    axs[3].set_title('Memory vs Epochs', fontsize=16)
    axs[3].set_yscale('log')
    axs[3].legend()
    axs[3].grid(True, alpha=0.4)

    plt.suptitle('Consumption of computational resources with epochs trained', fontsize=20)
    plt.tight_layout()
    plt.show()

    plt.savefig('resources_consumption.png', dpi=300, bbox_inches='tight')

def main():
    flop_b1_df = pd.read_csv("flops_profiler/training_flops_profiler_llama318_bs1.csv")
    flop_b2_df = pd.read_csv("flops_profiler/training_flops_profiler_llama318_bs2.csv")

    flop_b1 = flop_b1_df['FLOs']
    flop_b2 = flop_b2_df['FLOs']

    cuda_time_b1 = flop_b1_df['CUDA Time (ms)']
    cuda_time_b2 = flop_b2_df['CUDA Time (ms)']

    cpu_time_b1 = flop_b1_df['CPU Time (ms)']
    cpu_time_b2 = flop_b2_df['CPU Time (ms)']

    memory_b1 = flop_b1_df['Memory (MB)']
    memory_b2 = flop_b2_df['Memory (MB)']

    steps_b1 = flop_b1_df['Step']
    steps_b2 = flop_b2_df['Step']

    plot_accuracy_vs_flop(flop_b1, flop_b2, steps_b1, steps_b2)
    plot_resources_vs_flop(flop_b1, flop_b2, cuda_time_b1, cuda_time_b2, cpu_time_b1, cpu_time_b2, memory_b1, memory_b2, steps_b1, steps_b2)
    print("Plots saved as 'accuracy_vs_flos.png' and 'resources_consumption.png'.")


if __name__ == "__main__":
    main()