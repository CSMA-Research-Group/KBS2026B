"""Prepare only the records needed by a local analysis command."""
from pathlib import Path
import runpy
import sys
from archives import prepare

ROOT = Path(__file__).resolve().parents[2]
command = sys.argv[1]
arguments = sys.argv[2:]
commands = {
    'check':('check.py',None), 'test':('check.py',['language']),
    'verify':('check.py',[]), 'analyze-cpu':('cpu_reporting.py',[]),
    'replay-costs':('replay_costs.py',['language']), 'figures':('plot_figures.py',[]),
    'build':('build.py',[]),
}
if command == 'prepare':
    prepare(arguments or None)
    print('Data ready under work/evidence/')
else:
    script, groups = commands[command]
    if groups is None or groups:
        prepare(groups)
    if command == 'test': arguments = ['--tests-only',*arguments]
    if command == 'verify': arguments = ['--hashes-only',*arguments]
    sys.argv = [str(ROOT/'code/project'/script), *arguments]
    runpy.run_path(sys.argv[0], run_name='__main__')
