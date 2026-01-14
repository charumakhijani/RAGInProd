from __future__ import annotations
from contextlib import contextmanager
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

_tracer = None

def init_tracing(service_name: str, otlp_endpoint: str | None = None):
    global _tracer
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)

def tracer():
    return _tracer or trace.get_tracer("prod-rag")

@contextmanager
def span(name: str, **attrs):
    tr = tracer()
    with tr.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            sp.set_attribute(k, v)
        yield sp
