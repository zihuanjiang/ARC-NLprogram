# Lazy import to avoid loading arc.vis.report when running `python -m arc.vis.report`,
# which would trigger: "'arc.vis.report' found in sys.modules ... prior to execution"
__all__ = ["generate_pdf_report"]


def __getattr__(name: str):
    if name == "generate_pdf_report":
        from .report import generate_pdf_report
        return generate_pdf_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
