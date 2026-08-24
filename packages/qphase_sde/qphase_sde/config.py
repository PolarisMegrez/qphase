"""qphase_sde: Package Configuration (2.0)
---------------------------------------------------------
Resource-package-level configuration anchor of the SDE package.

Plugin configuration is owned by the plugin classes themselves: every plugin
declares its own ``config_schema`` (for example
``qphase_sde.analyser.psd.PsdAnalyzerConfig``), and the engine run
configuration lives with the engine (``qphase_sde.engine.EngineConfig``).
This module is deliberately minimal — it defines no configuration models yet;
genuinely package-level settings (resource profiles, package-wide defaults)
belong here when they are introduced.
"""
