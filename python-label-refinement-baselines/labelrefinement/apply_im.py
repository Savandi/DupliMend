import sys

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter


path = sys.argv[1]


def main():
    log = xes_importer.apply(path)

    net, initial_marking, final_marking = inductive_miner.apply(log,
                                                               variant=inductive_miner.Variants.IMf,
                                                               parameters={inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: 0.05})

    # for trace in log:
    #     for e in trace:
    #         if e['concept:name'] == 'S':
    #             e['concept:name'] = 'Open Expense Report'
    #         elif e['concept:name'] == 'A':
    #             e['concept:name'] = 'Fill out >=500€ Report'
    #         elif e['concept:name'] == 'B':
    #             e['concept:name'] = 'Send out Report'
    #         elif e['concept:name'] == 'C':
    #             e['concept:name'] = 'Receive Approval or Request'
    #         elif e['concept:name'] == 'D':
    #             e['concept:name'] = 'Fill out <500€ Report'
    #         elif e['concept:name'] == 'F':
    #             e['concept:name'] = 'Send out Report '
    #         elif e['concept:name'] == 'E':
    #             e['concept:name'] = 'Receive automatic Approval'
    #         elif e['concept:name'] == 'T':
    #             e['concept:name'] = 'Close Report'

    # net, initial_marking, final_marking = inductive_miner.apply(log)
    pnml_exporter.apply(net, initial_marking, f'road_traffic_fines_net_005.pnml', final_marking=final_marking)
    parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"}
    gviz_petri_net = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters)
    pn_visualizer.save(gviz_petri_net,
                       f'road_traffic_fines_net_005.png')


main()
