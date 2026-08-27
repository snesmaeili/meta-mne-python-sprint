"""Drive one browser through a sequence of actions, checking after every step.

Used by both the parametrized matrix and the random fuzzer, so a finding from
either reproduces the same way.
"""

import traceback
import warnings
from dataclasses import dataclass, field

import mne

from . import actions as _actions
from . import invariants
from .build import arm_tripwire, build


@dataclass
class Result:
    """What one run saw."""

    spec_label: str
    backend: str
    sequence: list = field(default_factory=list)
    violations: list = field(default_factory=list)  # (step, action, message)
    error: str = ""

    @property
    def ok(self):
        return not self.violations and not self.error

    def report(self):
        lines = [f"{self.spec_label} [{self.backend}]"]
        if self.error:
            lines.append(f"  ERROR after {len(self.sequence)} actions:\n{self.error}")
        for step, action, msg in self.violations:
            lines.append(f"  step {step} after {action!r}:")
            lines.extend(f"    {ln}" for ln in msg.splitlines())
        if self.sequence:
            lines.append(f"  sequence: {self.sequence}")
        return "\n".join(lines)


def close_fig(fig):
    """Close a browser on either backend.

    The Qt figure has ``close()``; matplotlib's ``MNEBrowseFigure`` is a
    ``Figure`` subclass and has no such method, so it goes through pyplot.
    """
    if fig is None:
        return
    closer = getattr(fig, "close", None)
    if callable(closer):
        try:
            closer()
            return
        except Exception:
            pass
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass


def open_browser(case, backend, **plot_kwargs):
    """Open a browser on a case with the padding tripwire armed."""
    mne.viz.set_browser_backend(backend)
    arm_tripwire(case.epochs)
    kwargs = dict(n_epochs=2, show=False)
    kwargs.update(plot_kwargs)
    fig = case.epochs.plot(**kwargs)
    fig.test_mode = True
    return fig


def run(spec, backend, names, *, plot_kwargs=None, strict_warnings=True):
    """Build, open, replay ``names``, and check invariants after every one.

    ``names`` are action names from :func:`actions.build_alphabet`; an unknown
    name is itself reported rather than skipped.
    """
    result = Result(spec_label=spec.label(), backend=backend)
    fig = None
    try:
        with warnings.catch_warnings():
            if strict_warnings:
                warnings.simplefilter("error")
                # matplotlib and pyqtgraph both chatter about offscreen drawing
                warnings.filterwarnings("ignore", message=".*non-interactive.*")
                warnings.filterwarnings("ignore", message=".*FigureCanvasAgg.*")
            case = build(spec)
            fig = open_browser(case, backend, **(plot_kwargs or {}))
            table = dict(
                _actions.build_alphabet(fig, case, backend, include_windows=True)
            )

            msgs = invariants.check_all(fig, case, backend)
            result.violations.extend((0, "<open>", m) for m in msgs)

            for step, name in enumerate(names, start=1):
                result.sequence.append(name)
                fn = table.get(name)
                if fn is None:
                    result.violations.append((step, name, f"unknown action {name!r}"))
                    continue
                fn()
                msgs = invariants.check_all(fig, case, backend)
                result.violations.extend((step, name, m) for m in msgs)
    except Exception:
        result.error = traceback.format_exc(limit=12)
    finally:
        if fig is not None:
            try:
                fig.close()
            except Exception:
                pass
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
    return result
