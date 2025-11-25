# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP
from minerl.herobraine.hero.handler import Handler
from typing import List

import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero import mc as MC  

TREECHOP_DOC = """
.. image:: ../assets/treechop1.mp4.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/treechop2.mp4.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/treechop3.mp4.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/treechop4.mp4.gif
  :scale: 100 %
  :alt:
In treechop, the agent must collect 64 `minercaft:log`. This replicates a common scenario in Minecraft, as logs are necessary to craft a large amount of items in the game, and are a key resource in Minecraft.

The agent begins in a forest biome (near many trees) with an iron axe for cutting trees. The agent is given +1 reward for obtaining each unit of wood, and the episode terminates once the agent obtains 64 units.
"""
TREECHOP_LENGTH = 8000
TREECHOP_WORLD_GENERATOR_OPTIONS = """{"coordinateScale":684.412,"heightScale":684.412,"lowerLimitScale":512.0,"upperLimitScale":512.0,"depthNoiseScaleX":200.0,"depthNoiseScaleZ":200.0,"depthNoiseScaleExponent":0.5,"mainNoiseScaleX":80.0,"mainNoiseScaleY":160.0,"mainNoiseScaleZ":80.0,"baseSize":8.5,"stretchY":12.0,"biomeDepthWeight":0.0,"biomeDepthOffset":0.0,"biomeScaleWeight":0.0,"biomeScaleOffset":0.0,"seaLevel":1,"useCaves":false,"useDungeons":false,"dungeonChance":8,"useStrongholds":false,"useVillages":false,"useMineShafts":false,"useTemples":false,"useMonuments":false,"useMansions":false,"useRavines":false,"useWaterLakes":false,"waterLakeChance":4,"useLavaLakes":false,"lavaLakeChance":80,"useLavaOceans":false,"fixedBiome":11,"biomeSize":4,"riverSize":1,"dirtSize":33,"dirtCount":10,"dirtMinHeight":0,"dirtMaxHeight":256,"gravelSize":33,"gravelCount":8,"gravelMinHeight":0,"gravelMaxHeight":256,"graniteSize":33,"graniteCount":10,"graniteMinHeight":0,"graniteMaxHeight":80,"dioriteSize":33,"dioriteCount":10,"dioriteMinHeight":0,"dioriteMaxHeight":80,"andesiteSize":33,"andesiteCount":10,"andesiteMinHeight":0,"andesiteMaxHeight":80,"coalSize":17,"coalCount":20,"coalMinHeight":0,"coalMaxHeight":128,"ironSize":9,"ironCount":20,"ironMinHeight":0,"ironMaxHeight":64,"goldSize":9,"goldCount":2,"goldMinHeight":0,"goldMaxHeight":32,"redstoneSize":8,"redstoneCount":8,"redstoneMinHeight":0,"redstoneMaxHeight":16,"diamondSize":8,"diamondCount":1,"diamondMinHeight":0,"diamondMaxHeight":16,"lapisSize":7,"lapisCount":1,"lapisCenterHeight":16,"lapisSpread":16}"""


class Treechop(HumanControlEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLTreechop-v0'

        super().__init__(*args,
                         max_episode_steps=TREECHOP_LENGTH, reward_threshold=64.0,
                         **kwargs)
        
    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForCollectingItems([
                dict(type="log", amount=1, reward=1.0),
            ])
        ]
    
    def create_agent_start(self) -> List[Handler]:
        return super().create_agent_start() + [
            handlers.SimpleInventoryAgentStart([
                dict(type="oak_log", quantity=3),
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromPossessingItem([
                dict(type="log", amount=64)]
            )
        ]
    
    def create_actionables(self):
        acts = [handlers.KeybasedCommandAction(action_name, key_binding)
                for key_binding, action_name in MC.KEYMAP.items()]
        acts.append(handlers.CameraAction())
        return acts
    
    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.DefaultWorldGenerator(force_reset="true",
                                           generator_options=TREECHOP_WORLD_GENERATOR_OPTIONS
                                           )
        ]
    
    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (TREECHOP_LENGTH * MS_PER_STEP)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=False
            )
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'survivaltreechop'

    def get_docstring(self):
        return TREECHOP_DOC


class ConfigurableTreechop(Treechop):
    """
    Treechop environment with configurable starting conditions for curriculum learning.
    
    Supports 3 independent switches that can be combined:
    - spawn_type: "random" or "near_tree" (near_tree not yet implemented)
    - with_logs: Number of starting logs (0-10)
    - with_axe: Whether to start with wooden axe
    
    Examples:
        # Easy - just needs to chop
        env = ConfigurableTreechop(with_axe=True).make()
        
        # Medium - has materials for axe
        env = ConfigurableTreechop(with_logs=6).make()
        
        # Hard - full task from scratch
        env = ConfigurableTreechop(with_logs=0, with_axe=False).make()
    """
    
    def __init__(
        self, 
        spawn_type: str = "random",
        with_logs: int = 0, 
        with_axe: bool = False,
        max_episode_steps: int = None,
        *args, 
        **kwargs
    ):
        """
        Args:
            spawn_type: "random" (default) or "near_tree" (not yet implemented)
            with_logs: Number of oak logs to start with (0-10)
            with_axe: Whether to start with a wooden axe equipped
            max_episode_steps: Override max steps (default: TREECHOP_LENGTH)
        """
        self.spawn_type = spawn_type
        self.with_logs = min(max(0, with_logs), 64)  # Clamp to valid range
        self.with_axe = with_axe
        
        # Set custom name based on config
        config_parts = []
        if with_axe:
            config_parts.append("axe")
        if with_logs > 0:
            config_parts.append(f"{with_logs}logs")
        if spawn_type != "random":
            config_parts.append(spawn_type)
        
        config_suffix = "_" + "_".join(config_parts) if config_parts else ""
        kwargs['name'] = f'MineRLConfigTreechop{config_suffix}-v0'
        
        # Override max episode steps if provided
        if max_episode_steps is not None:
            kwargs['max_episode_steps'] = max_episode_steps
        
        super().__init__(*args, **kwargs)
    
    def create_agent_start(self) -> List[Handler]:
        """Create agent start handlers with configured inventory."""
        # Get base handlers (excludes parent's inventory setup)
        base_handlers = Treechop.create_agent_start.__wrapped__(self) if hasattr(Treechop.create_agent_start, '__wrapped__') else []
        
        # Actually just use the grandparent's create_agent_start to avoid duplicate items
        from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
        base_handlers = HumanControlEnvSpec.create_agent_start(self)
        
        # Build custom inventory
        inventory = []
        
        if self.with_logs > 0:
            inventory.append(dict(type="oak_log", quantity=self.with_logs))
        
        if self.with_axe:
            inventory.append(dict(type="wooden_axe", quantity=1))
        
        if inventory:
            base_handlers.append(
                handlers.SimpleInventoryAgentStart(inventory)
            )
        
        # Handle spawn type (near_tree would need custom handler)
        if self.spawn_type == "near_tree":
            # TODO: Implement custom near-tree spawn handler
            # For now, forest biome (fixedBiome=11) provides naturally dense trees
            print("Warning: near_tree spawn not yet implemented, using random spawn")
        
        return base_handlers
    
    @classmethod
    def from_config(cls, config: dict) -> 'ConfigurableTreechop':
        """
        Create from config dict (e.g., from YAML).
        
        Args:
            config: Dict with keys 'spawn_type', 'with_logs', 'with_axe'
        
        Returns:
            ConfigurableTreechop instance
        """
        return cls(
            spawn_type=config.get('spawn_type', 'random'),
            with_logs=config.get('with_logs', 0),
            with_axe=config.get('with_axe', False),
            max_episode_steps=config.get('max_episode_steps', None)
        )


# Pre-defined curriculum stages for convenience
CURRICULUM_EASY = dict(spawn_type="random", with_logs=0, with_axe=True)
CURRICULUM_MEDIUM = dict(spawn_type="random", with_logs=6, with_axe=False)  
CURRICULUM_HARD = dict(spawn_type="random", with_logs=0, with_axe=False)