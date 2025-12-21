"""qphase: commands subpackage
---------------------------------------------------------
This subpackage contains the implementation of all CLI commands exposed by the
``qps`` tool, organized into logical groups using the Typer framework. It includes
commands for project initialization (``init``), job execution and monitoring
(``run``), plugin inspection and template generation (``plugin``), and system
configuration management (``config``). These commands serve as the primary user
interface, translating command-line arguments into API calls to the core components
while providing rich, color-coded terminal output.

Public API
----------
``run`` : Job execution commands (``qps run jobs``, ``qps run list``)
``plugin`` : Plugin commands (``qps list``, ``qps show``, ``qps template``)
``config`` : Configuration commands (``qps config show/set/reset``)
``init`` : Project initialization (``qps init``)
"""
