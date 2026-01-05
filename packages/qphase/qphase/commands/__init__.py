"""qphase: commands subpackage
---------------------------------------------------------
This subpackage contains the implementation of all CLI commands exposed by the
`qphase` tool, organized into logical groups using the Typer framework. It includes
commands for project initialization (`init`), job execution and monitoring
(`run`), plugin inspection and template generation (`list`, `show`, `template`),
and system configuration management (`config`). These commands serve as the primary user
interface, translating command-line arguments into API calls to the core components
while providing rich, color-coded terminal output.

Public API
----------
`run` : Job execution command (`qphase run`)
`list` : List plugins (`qphase list`)
`show` : Show plugin details (`qphase show`)
`template` : Generate plugin templates (`qphase template`)
`config` : Configuration commands (`qphase config show/set/reset`)
`init` : Project initialization (`qphase init`)
"""
