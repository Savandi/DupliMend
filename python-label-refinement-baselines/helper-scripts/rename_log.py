import json
import re
import sys
import string

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def main(args) -> None:
    print(args)
    # log_path = args[1]
    log_path = r'../real_logs/BPI Challenge 2017_3_cases_per_variant.xes.gz'
    log = xes_importer.apply(log_path)
    label_mapping = {}
    i = 0

    # for trace in log:
    #     for event in trace:
    #         event['concept:name'] = f"{event['concept:name']} {event['lifecycle:transition']}"

    # for trace in log:
    #     for event in trace:
    #         event['concept:name'] = f"{event['concept:name']} {event['lifecycle:transition']}"
    print('Before Convert')
    dataframe = pm4py.convert_to_dataframe(log)
    print('Converted')
    dataframe = dataframe[['case:concept:name', 'concept:name', 'time:timestamp']]
    log = pm4py.convert_to_event_log(dataframe)
    print('Converted back')

    for trace in log:
        for event in trace:
            label = event['concept:name']
            if label not in label_mapping:
                label_mapping[label] = string.printable[i]
                print(label)
                print(event)
                i += 1
            event['original_label'] = label
            event['concept:name'] = label_mapping[label]
            # print(label_mapping[label])

    pattern = r'/(.*)\.xes'
    print(log_path)
    match = re.match(pattern, log_path)

    # new_log_path = re.sub(match.group(1), f'{match.group(1)}_shortened_labels', log_path)
    new_log_path = '../real_logs/bpi_challenge_2017_3_cases_per_variant_shortened_labels.xes.gz'
    print(log)
    print(new_log_path)
    print(json.dumps(label_mapping))
    xes_exporter.apply(log, new_log_path)


def rename_from_short_to_original(args) -> None:
    print(args)
    # log_path = args[1]
    log_path = '../outputs/real_logs/caise_23_EVENTS_0_DistanceVariant.EDIT_DISTANCE_1_split_log.xes'
    log = xes_importer.apply(log_path)
    label_mapping = {}
    i = 0

    for trace in log:
        for event in trace:
            if event['concept:name'] == '3_0':
                event['concept:name'] = 'Send Report_0'
            elif event['concept:name'] == '3_1':
                event['concept:name'] = 'Send Report_1'
            else:
                event['concept:name'] = event['original_label']
            # print(label_mapping[label])

    pattern = r'/(.*)\.xes'
    print(log_path)
    match = re.match(pattern, log_path)

    # new_log_path = re.sub(match.group(1), f'{match.group(1)}_shortened_labels', log_path)
    new_log_path = '../real_logs/log_running_example_caise23_refined.xes'
    print(log)
    print(new_log_path)
    print(json.dumps(label_mapping))
    xes_exporter.apply(log, new_log_path)


main(sys.argv)
# rename_from_short_to_original(sys.argv)
