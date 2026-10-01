"""Compose optional inference without importing torch on the default CPU path."""

from ..domain.panda_rdf_safety import _query_pair


def build_pair_evaluator(bundle, pairs, options):
    if options.runtime_device == "cpu":
        return None
    from ..adapters.rdf_torch_runtime import TorchRDFPairs

    return TorchRDFPairs(
        bundle.models,
        bundle.queries,
        tuple(_query_pair(*pair) for pair in pairs),
        options,
        device=options.runtime_device,
    )
