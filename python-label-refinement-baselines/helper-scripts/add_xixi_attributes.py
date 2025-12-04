import sys

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer

log_path = str(sys.argv[1])
input_name = str(sys.argv[2])


def main():
    log = xes_importer.apply(log_path)

    for case in log:
        prev_label = ''
        for event in case:
            if prev_label == 'A':
                event['OrgLabel'] = 'F'
            else:
                event['OrgLabel'] = event['concept:name']
            prev_label = event['OrgLabel']

    xes_exporter.apply(log,
                       f'/home/jonas/repositories/pm-label-splitting/example_logs/{input_name}_LogD.xes')

    for case in log:
        for event in case:
            event['concept:name'] = event['OrgLabel']

    xes_exporter.apply(log,
                       f'/home/jonas/repositories/pm-label-splitting/example_logs/{input_name}_Log.xes')


main()
