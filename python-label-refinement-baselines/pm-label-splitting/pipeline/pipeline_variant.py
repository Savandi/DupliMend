import re
from enum import Enum


def remove_pipeline_variant_from_string(input_string: str) -> str:
    pattern = rf'\_({PipelineVariant.VARIANTS_MULTIPLEX}|{PipelineVariant.EVENTS_MULTIPLEX}|{PipelineVariant.EVENTS}|{PipelineVariant.VARIANTS})(_N_W)?'
    processed_input = re.sub(pattern, '', input_string)
    return processed_input


class PipelineVariant(Enum):
    EVENTS = 'EVENTS'
    EVENTS_MULTIPLEX = 'EVENTS_MULTIPLEX'
    VARIANTS = 'VARIANTS'
    VARIANTS_MULTIPLEX = 'VARIANTS_MULTIPLEX'

    def __str__(self):
        return str(self.value)
