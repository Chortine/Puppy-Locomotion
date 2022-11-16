from envs.wavego_flat_config import WavegoFlatCfg, WavegoFlatCfgPPO
from envs.a1_flat_config import A1FlatCfg, A1FlatCfgPPO
from envs.a1_flat_config_previous import A1FlatCfg as prev_A1FlatCfg
from envs.a1_flat_config_previous import A1FlatCfgPPO as prev_A1FlatCfgPPO

from legged_gym.envs.base.legged_robot import LeggedRobot
from envs.wavego_robot import WavegoRobot

# from legged_gym.utils.task_registry import task_registry
from rl_utils.task_registry import task_registry

task_registry.register("wavego_flat", WavegoRobot, WavegoFlatCfg(), WavegoFlatCfgPPO())
task_registry.register("a1_flat", WavegoRobot, A1FlatCfg(), A1FlatCfgPPO())


from legged_gym.envs.base.legged_robot import LeggedRobot
task_registry.register("a1_flat_prev", LeggedRobot, prev_A1FlatCfg(), prev_A1FlatCfgPPO())
