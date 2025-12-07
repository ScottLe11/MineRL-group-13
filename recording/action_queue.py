"""
Action Queue Manager for Discrete Action Recording.

Manages:
- Current action execution (with duration tracking)
- Single action queuing (buffer next action)
- Action state for UI display
"""

from typing import Dict, Optional, TYPE_CHECKING

# Avoid importing wrappers at module level to prevent cv2 dependency in tests
if TYPE_CHECKING:
    from wrappers.discrete_actions import DiscreteAction


class ActionQueue:
    """
    Manages discrete action execution with single-action queuing.

    Rules:
    - Only ONE action executes at a time
    - Human can queue ONE next action while current executes
    - Queued action starts immediately when current finishes
    - Attempts to queue when buffer full are rejected
    """

    def __init__(self, action_definitions: Dict[int, 'DiscreteAction']):
        """
        Initialize action queue.

        Args:
            action_definitions: Dict mapping action index to DiscreteAction
        """
        self.actions = action_definitions

        # Current action state
        self.current_action: Optional[int] = None
        self.current_remaining_steps: int = 0

        # Queue state (single slot)
        self.queued_action: Optional[int] = None

        # Statistics
        self.total_actions_executed = 0
        self.total_queue_rejections = 0

    def is_busy(self) -> bool:
        """Check if an action is currently executing."""
        return self.current_remaining_steps > 0

    def can_queue(self) -> bool:
        """Check if human can queue a new action."""
        return self.queued_action is None

    def queue_action(self, action_index: int) -> bool:
        """
        Attempt to queue an action.

        Args:
            action_index: Index of action to queue

        Returns:
            True if queued successfully, False if queue full
        """
        if action_index not in self.actions:
            raise ValueError(f"Invalid action index: {action_index}")

        # If queue is full, reject
        if not self.can_queue():
            self.total_queue_rejections += 1
            return False

        # If no current action, start immediately
        if not self.is_busy():
            self._start_action(action_index)
            return True

        # Otherwise, queue for later
        self.queued_action = action_index
        return True

    def _start_action(self, action_index: int):
        """Internal: Start executing an action."""
        action = self.actions[action_index]
        self.current_action = action_index
        self.current_remaining_steps = action.duration
        self.queued_action = None
        self.total_actions_executed += 1

    def step(self) -> int:
        """
        Execute one step of current action.

        Returns:
            Current action index to execute (0 if no action)
        """
        # No action executing, return noop
        if not self.is_busy():
            return 0

        # Get current action
        action_to_execute = self.current_action

        # Decrement remaining steps
        self.current_remaining_steps -= 1

        # Check if action finished
        if self.current_remaining_steps == 0:
            # Action complete
            if self.queued_action is not None:
                # Start queued action
                self._start_action(self.queued_action)
            else:
                # No queued action, go idle
                self.current_action = None

        return action_to_execute

    def clear(self):
        """Clear current action and queue (emergency stop)."""
        self.current_action = None
        self.current_remaining_steps = 0
        self.queued_action = None

    def get_state(self) -> dict:
        """
        Get current state for UI display.

        Returns:
            Dict with:
                - current_action: Action index or None
                - current_name: Action name or None
                - remaining_steps: Steps left for current action
                - queued_action: Queued action index or None
                - queued_name: Queued action name or None
                - is_busy: Whether action is executing
                - can_queue: Whether can queue new action
        """
        state = {
            'current_action': self.current_action,
            'current_name': None,
            'remaining_steps': self.current_remaining_steps,
            'queued_action': self.queued_action,
            'queued_name': None,
            'is_busy': self.is_busy(),
            'can_queue': self.can_queue(),
        }

        if self.current_action is not None:
            state['current_name'] = self.actions[self.current_action].display_name

        if self.queued_action is not None:
            state['queued_name'] = self.actions[self.queued_action].display_name

        return state

    def get_statistics(self) -> dict:
        """Get execution statistics."""
        return {
            'total_actions_executed': self.total_actions_executed,
            'total_queue_rejections': self.total_queue_rejections,
        }


if __name__ == "__main__":
    from wrappers.discrete_actions import DISCRETE_ACTION_POOL, get_enabled_actions

    print("✅ Action Queue Manager Test")

    # Use subset of actions
    enabled = get_enabled_actions([1, 8, 12, 15, 17, 25])
    queue = ActionQueue(enabled)

    print("\n1. Initial state:")
    state = queue.get_state()
    print(f"   Busy: {state['is_busy']}")
    print(f"   Can queue: {state['can_queue']}")

    print("\n2. Queue action 25 (attack_10, duration=10):")
    success = queue.queue_action(25)
    print(f"   Success: {success}")
    state = queue.get_state()
    print(f"   Current: {state['current_name']} ({state['remaining_steps']} steps left)")

    print("\n3. Queue action 1 (forward) while attack_10 executes:")
    success = queue.queue_action(1)
    print(f"   Success: {success}")
    state = queue.get_state()
    print(f"   Queued: {state['queued_name']}")

    print("\n4. Try to queue another action (should fail):")
    success = queue.queue_action(8)
    print(f"   Success: {success} (expected False)")

    print("\n5. Step through attack_10:")
    for i in range(10):
        action_idx = queue.step()
        state = queue.get_state()
        action_name = enabled[action_idx].name if action_idx in enabled else 'noop'
        print(f"   Step {i+1}: Execute {action_name:10s} (remaining: {state['remaining_steps']})")

    print("\n6. After attack_10, forward should start automatically:")
    state = queue.get_state()
    print(f"   Current: {state['current_name']}")
    print(f"   Queued: {state['queued_name']}")

    print("\n7. Statistics:")
    stats = queue.get_statistics()
    print(f"   Total actions: {stats['total_actions_executed']}")
    print(f"   Queue rejections: {stats['total_queue_rejections']}")

    print("\n✅ All tests passed!")
