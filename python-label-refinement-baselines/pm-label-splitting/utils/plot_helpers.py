from matplotlib import pyplot as plt


def plot_noise_to_f1_score(x_noise, y_unrefined, y_refined, input_name):
    plt.clf()
    plt.plot(x_noise, y_unrefined, 'bo', label='Unrefined Log')
    plt.plot(x_noise, y_refined, 'ro', label='Refined Log')
    plt.axis([0, 0.5, 0, 1])
    plt.xlabel('Noise Threshold')
    plt.ylabel('F1 Score')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f'/mnt/c/Users/Jonas/Desktop/pm-label-splitting/plots/{input_name}_noise_to_f1_score.png')