import csv
import glob
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
def main():
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    #rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    #rc('text', usetex=True)
    path = "../results/exp_2_prec_measurement_without_loops/*.csv"
    path_with_loops = "../results/exp_4_prec_measurement_with_loops/*.csv"

    #path = "../results/no_prec_measurement_without_loops/*.csv"
    #path_with_loops = "../results/no_prec_measurement_with_loops/*.csv"
    data = []
    data_l = []
    for fname in glob.glob(path):

        car_data = pd.read_csv(fname)
        data.append(car_data)

    data = pd.concat(data)

    for fname in glob.glob(path_with_loops):

        car_data = pd.read_csv(fname)
        data_l.append(car_data)

    data_l = pd.concat(data_l)

    data = extend_frame(data)
    data_l = extend_frame(data_l)

    print(data)
    print(data.mean())

    print(data[["improvement_original"]].describe())
    print(data_l[["improvement_original"]].describe())


    #evaluate_precision_improvements_to_imprecise2(data, data_l)
    evaluate_synergy(data, data_l)
    #evaluate_benefit_from_baseline(data, data_l)
    #evaluate_number_of_labels_bounded(data, data_l)
    #evaluate_runtime(data, data_l)

    plt.show()
    #evaluate_benefit_from_baseline(data, data_l)

    #pd_data.boxplot(column=['ref_log_normal_prec_bounded', "ref_log_comdec_prec_bounded", "ref_log_folding_prec_bounded",
    #                        "ref_log_semi_prec_bounded", "ref_log_no_vertical_prec_bounded", 'ref_log_all_prec_bounded'])

def evaluate_number_of_labels_bounded(data, data_l):
    columns = [ "real_num_of_labels_normal","real_num_of_labels_folding",
                 "real_num_of_labels_semi", "real_num_of_labels_no_vertical", "real_num_of_labels_comdec", ]


    labels = ["no extension", "folding",
                "semi-greedy", "omit intra ref.", "comunity det.", ]
    #columns = ['mapping_quality', 'mapping_folding_quality', 'mapping_semi_quality', 'mapping_folding_semi_quality']


    for i, row in data.iterrows():

        if row["imp_prec"] < row["ref_log_prec"]:
            data.at[i, "real_num_of_labels_normal"] = row["num_of_new_labels"]
        else:
            data.at[i, "real_num_of_labels_normal"] = 1

        if row["imp_prec"] < row["ref_log_comdec_prec"]:
            data.at[i, "real_num_of_labels_comdec"] = row["num_of_new_labels_comdec"]
        else:
            data.at[i, "real_num_of_labels_comdec"] = 1

        if row["imp_prec"] < row["ref_log_folding_prec"]:
            data.at[i, "real_num_of_labels_folding"] = row["num_of_new_labels_folding"]
        else:
            data.at[i, "real_num_of_labels_folding"] = 1

        if row["imp_prec"] < row["ref_log_semi_prec"]:
            data.at[i, "real_num_of_labels_semi"] = row["num_of_new_labels_semi"]
        else:
            data.at[i, "real_num_of_labels_semi"] = 1

        if row["imp_prec"] < row["ref_log_no_vertical_prec"]:
            data.at[i, "real_num_of_labels_no_vertical"] = row["num_of_new_labels_no_vertical"]
        else:
            data.at[i, "real_num_of_labels_no_vertical"] = 1


    for i, row in data_l.iterrows():

        if row["imp_prec"] < row["ref_log_prec"]:
            data_l.at[i, "real_num_of_labels_normal"] = row["num_of_new_labels"]
        else:
            data_l.at[i, "real_num_of_labels_normal"] = 1

        if row["imp_prec"] < row["ref_log_comdec_prec"]:
            data_l.at[i, "real_num_of_labels_comdec"] = row["num_of_new_labels_comdec"]
        else:
            data_l.at[i, "real_num_of_labels_comdec"] = 1

        if row["imp_prec"] < row["ref_log_folding_prec"]:
            data_l.at[i, "real_num_of_labels_folding"] = row["num_of_new_labels_folding"]
        else:
            data_l.at[i, "real_num_of_labels_folding"] = 1

        if row["imp_prec"] < row["ref_log_semi_prec"]:
            data_l.at[i, "real_num_of_labels_semi"] = row["num_of_new_labels_semi"]
        else:
            data_l.at[i, "real_num_of_labels_semi"] = 1

        if row["imp_prec"] < row["ref_log_no_vertical_prec"]:
            data_l.at[i, "real_num_of_labels_no_vertical"] = row["num_of_new_labels_no_vertical"]
        else:
            data_l.at[i, "real_num_of_labels_no_vertical"] = 1


    means = data[columns].mean()
    means_l = data_l[columns].mean()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width / 2, means, width, label='No imp.t. in l.')
    rects2 = ax.barh(x + width / 2, means_l, width, label='Imp. t. in l.')

    # rects2[1].set_color("r")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Extension')
    ax.set_xlabel('Average number of new labels')
    #ax.set_title('Mean improvement for combinations of extensions and precise refinement.')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.legend()

    ax.invert_yaxis()
    for index, value in enumerate(means_l):
        plt.text(value, index + 0.35, " " + str(round(value, 2)))

    for index, value in enumerate(means):
        plt.text(value, index, " " + str(round(value, 2)))

    fig.tight_layout()
    fig.show()
    fig.savefig("../results/evaluate_number_of_labels_bounded.pdf", bbox_inches='tight')

def evaluate_number_of_labels(data, data_l):
    columns = ["num_of_new_labels",
               "num_of_new_labels_comdec", "num_of_new_labels_folding", "num_of_new_labels_semi", "num_of_new_labels_no_vertical"]
    #columns = ['mapping_quality', 'mapping_folding_quality', 'mapping_semi_quality', 'mapping_folding_semi_quality']
    labels = columns

    means = data[columns].mean()
    means_l = data_l[columns].mean()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width / 2, means, width, label='Imprecise task in loop')
    rects2 = ax.barh(x + width / 2, means_l, width, label='No imprecise task in loop')

    # rects2[1].set_color("r")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Refinement mode')
    ax.set_xlabel('Increase in log precision  (\%)')
    ax.set_title('Mean improvement for combinations of extensions and precise refinement.')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.legend()

    ax.invert_yaxis()
    for index, value in enumerate(means_l):
        plt.text(value, index + 0.35, " " + str(round(value, 3)))

    for index, value in enumerate(means):
        plt.text(value, index, " " + str(round(value, 3)))

    fig.tight_layout()
    fig.show()
    fig.savefig("../results/evaluate_number_of_labels.pdf", bbox_inches='tight')
def evaluate_runtime(data, data_l):

    data = data[['time_for_greedy_mapping', 'time_for_semi_greedy_mapping']]
    data_l = data_l[['time_for_greedy_mapping', 'time_for_semi_greedy_mapping']]


    print("no loop")
    print(data.describe())
    print("loop")
    print(data_l.describe())

    fig, ax = plt.subplots()
    labels = ["No imprecise tasks in loops", "Imprecise tasks in loops"]
    import seaborn as sns

    data1 = pd.DataFrame(data, columns=['time_for_greedy_mapping', 'time_for_semi_greedy_mapping']).assign(Location="No imprecise tasks in loops")
    data2 = pd.DataFrame(data_l + 0.2, columns=['time_for_greedy_mapping', 'time_for_semi_greedy_mapping']).assign(Location="Imprecise tasks in loops")

    data1.rename(columns={'time_for_greedy_mapping': 'Greedy mapping search', "time_for_semi_greedy_mapping": "Semi-greedy mapping search"}, inplace=True)
    data2.rename(columns={'time_for_greedy_mapping': 'Greedy mapping search', "time_for_semi_greedy_mapping": "Semi-greedy mapping search"}, inplace=True)

    #data3 = pd.DataFrame(np.random.rand(17, 3) + 0.4, columns=['A', 'B', 'C']).assign(Location=3)

    cdf = pd.concat([data1, data2])
    mdf = pd.melt(cdf, id_vars=['Location'], var_name=['Letter'])
    #print(mdf.head())

    ax = sns.boxplot(x="Location", y="value", hue="Letter", data=mdf, showfliers=False, showmeans=True, meanprops={"marker":"s","markerfacecolor":"red", "markeredgecolor":"black"})
    #print(data["time_for_greedy_mapping"])
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("")
    ax.legend(loc="center right")

    fig.show()
    fig.savefig("../results/evaluate_number_of_labels.pdf", bbox_inches='tight')
    ax.set_xticklabels(labels)
    fig.show()
    fig.savefig("../results/evaluate_runtime_new.pdf", bbox_inches='tight')
    ################
        #data.boxplot(column=['time_for_greedy_mapping', "time_for_semi_greedy_mapping"], showfliers=False)
    ###########plt.boxplot([data_a, data_b, data_l_a, data_l_b])
    #data.boxplot(column=['time_for_greedy_mapping', "time_for_semi_greedy_mapping"], showfliers=False)
    #data_l.boxplot(column=['time_for_greedy_mapping', "time_for_semi_greedy_mapping"], showfliers=False)
    #boxplot with and without outliers
    #because outliers ---> greedy, if number of equal labels extends specific value (short loop with many equal iterations makes trouble


def evaluate_modularity(data, data_l):
    data_l.boxplot(column=['mapping_quality', 'mapping_folding_quality', 'mapping_semi_quality', 'mapping_folding_semi_quality']).set_ylim([0.7, 0.9])
    print(data[['mapping_quality', 'mapping_folding_quality']].describe())
    print(stats.ttest_rel(data_l['mapping_quality'], data_l['mapping_folding_quality']))

#####################################

def evaluate_precision_improvements_to_imprecise2(data, data_l):
    columns = ["improvement_normal",
          "improvement_folding", "improvement_no_vertical", "improvement_semi", "improvement_comdec",
          "improvement_no_comdec", "improvement_no_semi", "improvement_vertical", "improvement_no_folding",
               "improvement_all", "improvement_original"]

    labels = columns
    labels = ["no extension",
               "folding", "omit intra ref.", "semi-greedy", "community det.",
              "no community det.", "no semi-greedy", "no omit intra ref.", "no folding",
              "all extensions", "original refinement *"]
    means = data[columns].mean() * 100
    means_l = data_l[columns].mean() * 100


    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width / 2, means, width, label='No imprecise task in loop')
    rects2 = ax.barh(x + width / 2, means_l, width, label='Imprecise task in loop')


    #rects2[1].set_color("r")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Extension')
    ax.set_xlabel('Average increase in log precision (\%)')
    #ax.set_title('Mean improvement for combinations of extensions and precise refinement.')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.legend()

    ax.invert_yaxis()
    for index, value in enumerate(means_l):
        plt.text(value, index+0.35, " " + str(round(value, 2)))

    for index, value in enumerate(means):
        plt.text(value, index, " " + str(round(value, 2)))

    fig.tight_layout()
    fig.show()
    fig.savefig("../results/evaluate_precision_improvements_to_imprecise.pdf", bbox_inches='tight')

def evaluate_precision_improvements_to_imprecise(data, data_l):
    #plt.rcdefaults()
    #data.append(data_l)
    #data = data_l
    columns = ["improvement_normal",
          "improvement_folding", "improvement_no_vertical", "improvement_semi", "improvement_comdec",
          "improvement_no_comdec", "improvement_no_semi", "improvement_vertical", "improvement_no_folding", "improvement_all", "improvement_original"]


    #n_bins = 15
    #ax.hist(data[columns], n_bins, histtype='step', stacked=True, fill=False)
    #ax.set_title('stack step (unfilled)')




    means = data[columns].mean() * 100
    means_l = data_l[columns].mean() * 100

    #means.sort_values(axis=0)
    #sem = data[columns].sem() * 100

    #print("standard deviation:")
    #print(sem)



    #fig.autolayout: True

    y_pos = np.arange(len(columns))

    labels = columns
    #labels = ["normal",
    #          "community det.", "folding", "semi-greedy", "omit inside ref.",
    #          "no community det.", "no folding", "no semi-greedy", "no omit inside ref.", "all extensions", "improvement_original"]

    #df = pd.DataFrame({"labels": labels, "means": means, "sem": sem})
    ##df = df.sort_values("means")
    #labels = df["labels"]
    #means = df["means"]
    #sem = df["sem"]

    ####good:

    df = pd.DataFrame({"Imprecise Label in Loop": means_l.T, "No imprecise label in Loop": (abs(means - means_l)).T})
    #df = pd.DataFrame([means_l, means - means_l]).T

    ax = df.plot.barh(stacked=True, label=labels)
    #ax.patches[df.index.get_indexer(['A'])[0]].set_facecolor('r')


    ax.invert_yaxis()
    ax.set_yticklabels(labels)


    for index, value in enumerate(means_l):
        plt.text(value, index+0.1, " "+ str(round(value, 2)))

    for index, value in enumerate(means):
        plt.text(value, index+0.1, "   "+ str(round(value, 2)))

def evaluate_precision_improvements_to_normal_histogram(data, data_l):

    columns = ["comdec_vs_normal",
          "folding_vs_normal", "semi_vs_normal", "no_vertical_vs_normal", "no_comdec_vs_normal",
          "no_folding_vs_normal", "no_semi_vs_normal", "vertical_vs_normal", "all_vs_normal"]
    ...
    #imp, normal
    #normal, ext1,.... noext1, ....., optimal
    #1. histogram


def evaluate_precision_improvements_to_normal_table(data, data_l):
    ...
    #normal, ext1,.... noext1, ....., optimal
    #1. histogram

def evaluate_precision_improvements_to_normal_paired_ttest(data, data_l):
    ...
    #table with all variables


def evaluate_benefit_from_baseline(data, data_l):
    data = data.append(data_l)

    columns = ["normal_gain_bounded",
               "comdec_gain_bounded", "folding_gain_bounded", "semi_gain_bounded",  "omitvertical_gain_bounded",
               "no_comdec_gain_bounded", "no_folding_gain_bounded", "no_semi_gain_bounded", "no_omitvertical_gain_bounded",
               "all_gain_bounded"]

    labels = ["no extension",
               "community det.", "folding", "semi-greedy",  "omit intra ref.",
               "no community det.", "no folding", "no semi-greedy", "no omit intra ref.",
               "all"]
    #labels = ["normal",
    #          "folding", "omit inside ref.", "semi-greedy", "community det.",
    #          "no community det.", "no semi-greedy", "no omit inside ref.", "no folding",
    #          "all extensions", "precise refinement"]
    means = data[columns].mean() * 100
    means_l = data_l[columns].mean() * 100


    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width / 2, means, width, label='No imprecise task in loop')
    rects2 = ax.barh(x + width / 2, means_l, width, label='Imprecise task in loop')


    #rects2[1].set_color("r")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Extension')
    ax.set_xlabel('Benefit from bound (\%)')
    #ax.set_title('Difference between bounded log precision and log precision of event log refined with given method.')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.legend()

    ax.invert_yaxis()
    for index, value in enumerate(means_l):
        plt.text(value, index+0.35, " " + str(round(value, 2)))

    for index, value in enumerate(means):
        plt.text(value, index, " " + str(round(value, 2)))

    fig.tight_layout()
    fig.show()
    fig.savefig("../results/evaluate_benefit_from_baseline.pdf", bbox_inches='tight')
    #improvement when single used VS improvement when used with all others


def evaluate_synergy(data, data_l):
    columns = ["folding_synergy", "omitvertical_synergy", "semi_synergy",  "comdec_synergy", "all_synergy"]

    labels = columns
    labels = ["folding", "omit intra ref.", "semi-greedy",  "community det.", "all"]
    #labels = ["normal",
    #          "folding", "omit inside ref.", "semi-greedy", "community det.",
    #          "no community det.", "no semi-greedy", "no omit inside ref.", "no folding",
    #          "all extensions", "precise refinement"]
    means = data[columns].mean() * 100
    means_l = data_l[columns].mean() * 100


    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.barh(x - width / 2, means, width, label='No imprecise task in loop')
    rects2 = ax.barh(x + width / 2, means_l, width, label='Imprecise task in loop')


    #rects2[1].set_color("r")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Extension')
    ax.set_xlabel('Average log precision synergy (\%)')
    #ax.set_title('Synergy of single extensions and all extensions combined.')
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.legend()

    ax.invert_yaxis()
    for index, value in enumerate(means_l):
        plt.text(value, index+0.35, " " + str(round(value, 2)))

    for index, value in enumerate(means):
        plt.text(value, index, " " + str(round(value, 2)))

    fig.tight_layout()

    fig.show()

    fig.savefig("../results/evaluate_synergy.pdf", bbox_inches='tight')
    #improvement when single used VS improvement when used with all others

def evaluate_help_of_backup_plan(data, data_l):
    ...
    #ala: comdec makes more better but also more worse refinements

    ...

def extend_frame(pd_data):
    pd_data['ref_log_normal_prec_bounded'] = pd_data[['ref_log_prec', 'imp_prec']].max(axis=1)

    pd_data['ref_log_comdec_prec_bounded'] = pd_data[['ref_log_comdec_prec', 'imp_prec']].max(axis=1)
    pd_data['ref_log_folding_prec_bounded'] = pd_data[['ref_log_folding_prec', 'imp_prec']].max(axis=1)
    pd_data['ref_log_semi_prec_bounded'] = pd_data[['ref_log_semi_prec', 'imp_prec']].max(axis=1)
    pd_data['ref_log_no_vertical_prec_bounded'] = pd_data[['ref_log_no_vertical_prec', 'imp_prec']].max(axis=1)

    pd_data['ref_log_no_comdec_prec_bounded'] = pd_data[['ref_log_no_comdec_prec', 'imp_prec']].max(axis=1)
    pd_data['ref_log_no_folding_prec_bounded'] = pd_data[['ref_log_no_folding_prec', 'imp_prec']].max(axis=1)
    pd_data['ref_log_no_semi_prec_bounded'] = pd_data[['ref_log_no_semi_prec', 'imp_prec']].max(axis=1)
    pd_data['ref_log_vertical_prec_bounded'] = pd_data[['ref_log_vertical_prec', 'imp_prec']].max(axis=1)

    pd_data['ref_log_all_prec_bounded'] = pd_data[['ref_log_all_prec', 'imp_prec']].max(axis=1)


    pd_data["normal_gain_bounded"] = (pd_data["ref_log_normal_prec_bounded"] - pd_data["ref_log_prec"])
    pd_data["comdec_gain_bounded"] = (pd_data["ref_log_comdec_prec_bounded"] - pd_data["ref_log_comdec_prec"])
    pd_data["folding_gain_bounded"] = (pd_data["ref_log_folding_prec_bounded"] - pd_data["ref_log_folding_prec"])
    pd_data["semi_gain_bounded"] = (pd_data["ref_log_semi_prec_bounded"] - pd_data["ref_log_semi_prec"])
    pd_data["omitvertical_gain_bounded"] = (pd_data["ref_log_no_vertical_prec_bounded"] - pd_data["ref_log_no_vertical_prec"])

    pd_data["no_comdec_gain_bounded"] = (pd_data["ref_log_no_comdec_prec_bounded"] - pd_data["ref_log_no_comdec_prec"])
    pd_data["no_folding_gain_bounded"] = (pd_data["ref_log_no_folding_prec_bounded"] - pd_data["ref_log_no_folding_prec"])
    pd_data["no_semi_gain_bounded"] = (pd_data["ref_log_no_semi_prec_bounded"] - pd_data["ref_log_no_semi_prec"])
    pd_data["no_omitvertical_gain_bounded"] = (pd_data["ref_log_vertical_prec_bounded"] - pd_data["ref_log_vertical_prec"])
    pd_data["all_gain_bounded"] = (pd_data["ref_log_all_prec_bounded"] - pd_data["ref_log_all_prec"])


    ############################################################

    pd_data['comdec_vs_normal'] = (pd_data['ref_log_comdec_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])
    pd_data['folding_vs_normal'] = (pd_data['ref_log_folding_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])
    pd_data['semi_vs_normal'] = (pd_data['ref_log_semi_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])
    pd_data['no_vertical_vs_normal'] = (pd_data['ref_log_no_vertical_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])

    pd_data['no_comdec_vs_normal'] = (pd_data['ref_log_no_comdec_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])
    pd_data['no_folding_vs_normal'] = (pd_data['ref_log_no_folding_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])
    pd_data['no_semi_vs_normal'] = (pd_data['ref_log_no_semi_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])
    pd_data['vertical_vs_normal'] = (pd_data['ref_log_vertical_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])

    pd_data['all_vs_normal'] = (pd_data['ref_log_all_prec_bounded'] - pd_data['ref_log_normal_prec_bounded'])

#####################################################
    pd_data["improvement_normal"] = pd_data["ref_log_normal_prec_bounded"] - pd_data["imp_prec"]
    pd_data["improvement_original"] = pd_data["precise_refined_log_prec"] - pd_data["imp_prec"]

    pd_data["improvement_normal"] = pd_data["ref_log_normal_prec_bounded"] - pd_data["imp_prec"]
    pd_data["improvement_comdec"] = pd_data["ref_log_comdec_prec_bounded"] - pd_data["imp_prec"]
    pd_data["improvement_folding"] = pd_data["ref_log_folding_prec_bounded"] - pd_data["imp_prec"]
    pd_data["improvement_semi"] = pd_data["ref_log_semi_prec_bounded"] - pd_data["imp_prec"]
    pd_data["improvement_no_vertical"] = pd_data["ref_log_no_vertical_prec_bounded"] - pd_data["imp_prec"]

    pd_data["improvement_no_comdec"] = pd_data["ref_log_no_comdec_prec_bounded"] - pd_data["imp_prec"]
    pd_data["improvement_no_folding"] = pd_data["ref_log_no_folding_prec_bounded"] - pd_data["imp_prec"]
    pd_data["improvement_no_semi"] = pd_data["ref_log_no_semi_prec_bounded"] - pd_data["imp_prec"]
    pd_data["improvement_vertical"] = pd_data["ref_log_vertical_prec_bounded"] - pd_data["imp_prec"]

    pd_data["improvement_all"] = pd_data["ref_log_all_prec_bounded"] - pd_data["imp_prec"]

    pd_data["comdec_synergy"] = (pd_data["improvement_all"] - pd_data["improvement_no_comdec"]) - (pd_data["improvement_comdec"] - pd_data["improvement_normal"])
    pd_data["folding_synergy"] = (pd_data["improvement_all"] - pd_data["improvement_no_folding"]) - (pd_data["improvement_folding"] - pd_data["improvement_normal"])
    pd_data["semi_synergy"] = (pd_data["improvement_all"] - pd_data["improvement_no_semi"]) - (pd_data["improvement_semi"] - pd_data["improvement_normal"])
    pd_data["omitvertical_synergy"] = (pd_data["improvement_all"] - pd_data["improvement_vertical"]) - (pd_data["improvement_no_vertical"] - pd_data["improvement_normal"])

    pd_data["all_synergy"] = (pd_data["improvement_all"] - pd_data["improvement_normal"]) \
                                - (pd_data["improvement_comdec"] - pd_data["improvement_normal"])\
                                - (pd_data["improvement_folding"] - pd_data["improvement_normal"])\
                                - (pd_data["improvement_semi"] - pd_data["improvement_normal"])\
                                - (pd_data["improvement_no_vertical"] - pd_data["improvement_normal"])




    return pd_data
    ...




def get_data_from_file(fname):
    with open(fname) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = [row for row in reader]

    #print(data_read)
    return data_read

main()