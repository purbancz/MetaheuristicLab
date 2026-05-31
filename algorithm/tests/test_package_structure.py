from algorithm.AdaptivePSO import CoAdaptativePSO as LegacyCoAdaptativePSO
from algorithm.CMAES import CMAES as LegacyCMAES
from algorithm.LSHADE import LSHADE as LegacyLSHADE
from algorithm.PGCHEA import PGCHEA as LegacyPGCHEA
from algorithm.WAPSO import WorstAwarePSO as LegacyWorstAwarePSO
from algorithm.hybrid_diverse import HybridAdditivePSO as LegacyHybridAdditivePSO
from algorithm.particles_with_roles import RebelPSO as LegacyRebelPSO
from algorithm.reinitialized_PSO import FRAPSO as LegacyFRAPSO
from algorithm.single_objective_PSO import SingleObjectivePSO as LegacySingleObjectivePSO

from algorithm.basic import CoAdaptativePSO, FRAPSO, SingleObjectivePSO, WorstAwarePSO
from algorithm.role_based.role_hybrids import HybridAdditivePSO
from algorithm.pso_ga_hybrids import PGCHEA
from algorithm.role_based.roles import RebelPSO
from algorithm.sota import CMAES, LSHADE


def test_new_package_exports_match_legacy_import_paths() -> None:
    assert LegacySingleObjectivePSO is SingleObjectivePSO
    assert LegacyWorstAwarePSO is WorstAwarePSO
    assert LegacyCoAdaptativePSO is CoAdaptativePSO
    assert LegacyFRAPSO is FRAPSO
    assert LegacyRebelPSO is RebelPSO
    assert LegacyHybridAdditivePSO is HybridAdditivePSO
    assert LegacyPGCHEA is PGCHEA
    assert LegacyCMAES is CMAES
    assert LegacyLSHADE is LSHADE
