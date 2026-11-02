"""Resource planning for NEW workloads; never changes a frozen experiment."""
import math
import os
from pathlib import Path

from innovation_stage3.storage import ROOT, read


def host_snapshot():
    allowed = sorted(os.sched_getaffinity(0))
    groups = {}
    for cpu in allowed:
        topology = Path('/sys/devices/system/cpu') / ('cpu' + str(cpu)) / 'topology'
        try:
            key = (int((topology / 'physical_package_id').read_text()), int((topology / 'core_id').read_text()))
        except (FileNotFoundError, ValueError):
            key = (0, cpu)
        groups.setdefault(key, []).append(cpu)
    memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    quotas, memory_headrooms = [], []
    # Respect quotas when readable, including parent cgroup limits.
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers == '':
            base = Path('/sys/fs/cgroup')
            position = base / relative.lstrip('/')
            while position.is_relative_to(base):
                p = position / 'cpu.max'
                if p.exists():
                    first, period = p.read_text().split()
                    if first != 'max':
                        quotas.append(int(first) / int(period))
                limit, used = position / 'memory.max', position / 'memory.current'
                if limit.exists() and used.exists() and limit.read_text().strip() != 'max':
                    memory_headrooms.append(max(0, int(limit.read_text()) - int(used.read_text())) // 1048576)
                if position == base:
                    break
                position = position.parent
        elif 'cpu' in controllers.split(','):
            for base in (Path('/sys/fs/cgroup/cpu'), Path('/sys/fs/cgroup/cpu,cpuacct')):
                position = base / relative.lstrip('/')
                while position.is_relative_to(base):
                    p, period = position / 'cpu.cfs_quota_us', position / 'cpu.cfs_period_us'
                    if p.exists() and period.exists() and int(p.read_text()) > 0:
                        quotas.append(int(p.read_text()) / int(period.read_text()))
                    if position == base:
                        break
                    position = position.parent
        elif 'memory' in controllers.split(','):
            base = Path('/sys/fs/cgroup/memory')
            position = base / relative.lstrip('/')
            while position.is_relative_to(base):
                limit, used = position / 'memory.limit_in_bytes', position / 'memory.usage_in_bytes'
                if limit.exists() and used.exists():
                    memory_headrooms.append(max(0, int(limit.read_text()) - int(used.read_text())) // 1048576)
                if position == base:
                    break
                position = position.parent
    return {'logical_cpus': allowed, 'physical_core_groups': [v for _, v in sorted(groups.items())],
            'physical_core_count': len(groups), 'memory_available_mib': memory['MemAvailable'] // 1024,
            'memory_total_mib': memory['MemTotal'] // 1024, 'cpu_quota_cores': min(quotas) if quotas else None,
            'cgroup_memory_available_mib': min(memory_headrooms) if memory_headrooms else None,
            'load_average': list(os.getloadavg())}


def choose(snapshot, config, jobs, mode='throughput'):
    if mode not in ('throughput', 'physical', 'timing') or jobs < 1:
        raise ValueError('Invalid workload/resource mode')
    groups = snapshot['physical_core_groups']
    reserve = min(config['reserve_physical_cores'], max(0, len(groups) - 1))
    worker_groups = groups[:-reserve] if reserve else groups
    reserved = [c for g in groups[-reserve:] for c in g] if reserve else []
    cpus = [c for g in worker_groups for c in g] if mode == 'throughput' else [g[0] for g in worker_groups]
    if mode == 'timing':
        cpus = [(reserved or cpus)[-1]]
    available_memory = min(snapshot['memory_available_mib'], snapshot.get('cgroup_memory_available_mib')
                           if snapshot.get('cgroup_memory_available_mib') is not None else snapshot['memory_available_mib'])
    memory_slots = (available_memory - config['reserve_memory_mib']) // config['worker_address_space_mib']
    if memory_slots < 1:
        raise RuntimeError('Insufficient available memory after reserve')
    quota_slots = math.floor(snapshot['cpu_quota_cores']) if snapshot['cpu_quota_cores'] is not None else len(cpus)
    workers = min(jobs, len(cpus), memory_slots, max(1, quota_slots))
    return {'mode': mode, 'workers': workers, 'worker_cpus': cpus[:workers], 'reserved_cpus': reserved,
            'worker_address_space_mib': config['worker_address_space_mib'], 'nice': config['throughput_nice'],
            'thread_libraries_per_worker': 1, 'gpu_requests': 0, 'host': snapshot,
            'timing_results': 'not_measured' if mode != 'timing' else 'require_validated_clocks_and_isolation'}


def plan_resources(jobs, mode='throughput'):
    return choose(host_snapshot(), read(ROOT / 'code/configs/experiment_resources.json'), jobs, mode)
