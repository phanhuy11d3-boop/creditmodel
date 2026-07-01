"""Production-grade Probability-of-Default (PD) application scorecard.

Public convention (kept consistent end-to-end):

* Target column ``default``: ``1 == Bad`` (the modeled event), ``0 == Good``.
* WoE orientation: ``WoE = ln(%Good / %Bad)`` so positive WoE == better applicant.
* The model estimates ``P(Bad)``; higher TotalScore <=> lower PD.

See ``docs``/MDD for the full orientation-consistency chain.
"""

__version__ = "0.1.0"
