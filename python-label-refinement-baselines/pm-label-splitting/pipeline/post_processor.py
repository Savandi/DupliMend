from typing import Dict

from pm4py.objects.petri_net.obj import PetriNet


class PostProcessor:
    """
    Post-processor, responsible for the last step of the pipeline.
    Renames the temporary labels to the labels used for evaluation of the model and renames shortened temporary labels
    to the original labels.
    """

    def __init__(self, split_labels_to_original_labels: Dict[str, str],
                 short_labels_to_original_labels: Dict[str, str]):
        self._split_labels_to_original_labels = split_labels_to_original_labels
        self._short_labels_to_original_labels = short_labels_to_original_labels

    def post_process_petri_net(self, net: PetriNet) -> PetriNet:
        """
        Renames split labels from X_1, X_2,... to X to enable evaluation

        :param net: Raw petri net mined from split event log
        :return: Post-processed petri net
        """
        for transition in net.transitions:
            if transition.label is not None and transition.label[0] in self._split_labels_to_original_labels.values():
                transition.label = transition.label[0]
        return net

    def rename_short_labels_to_original_labels(self, net: PetriNet) -> PetriNet:
        """
        Rename shortened labels used to make distance calculation easier to original label

        :param net: Post-processed petri net
        :return: Petri net with original label names
        """
        if self._short_labels_to_original_labels is None:
            print('No mapping from short labels to original labels')
            return net
        for transition in net.transitions:
            if transition.label in self._short_labels_to_original_labels.keys():
                transition.label = self._short_labels_to_original_labels[transition.label]
        return net
