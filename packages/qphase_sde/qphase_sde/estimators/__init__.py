"""qphase_sde: Analyser Support Plugins
---------------------------------------------------------
The plugin families that an analyser delegates its estimation work to. They
are registered plugins in their own namespaces, but they are never run by the
scheduler directly: an analyser resolves one child per job and calls it.

Public API
----------
spectral_estimator
    PSD estimators (periodogram, Welch, multitaper).
peak_finder
    Candidate peak finders over a spectrum.
peak_tracker
    Path trackers that link peak candidates across scan points.
coherence_frequency
    Coherence-to-frequency estimators (short delay, band limited,
    finite delay).
"""
