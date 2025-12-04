from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter


class LogGenerator:
    """
    Generate log from petri net
    """

    def __init__(self):
        pass

    def get_log_from_bpmn(self, bpmn_graph: BPMN) -> EventLog:
        """
        Generate log from petri net

        :param bpmn_graph: BPMN Model
        :return: Simulated event log
        """
        net, im, fm = bpmn_converter.apply(bpmn_graph)
        pnml_exporter.apply(net, im,
                             f'/home/jonas/repositories/pm-label-splitting/example_logs/daily_life_example_net.pnml',
                             final_marking=fm)
        event_log = simulator.apply(net, im, variant=simulator.Variants.BASIC_PLAYOUT,
                                    parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 200})
        num_cases = len(event_log)
        print(num_cases)
        return event_log
