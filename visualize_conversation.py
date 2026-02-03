#!/usr/bin/env python3
"""
Script to visualize multi-turn conversations from detailed log files.
Usage: python visualize_conversation.py <task_id> <domain> [trial]
  - task_id: The task ID to visualize (e.g., "0", "1", etc.)
  - domain: Either "retail" or "airline"
  - trial: The trial number (default: 0)
"""

import json
import sys
from pathlib import Path


def format_tool_calls(tool_calls):
    """Format tool calls for display."""
    if not tool_calls:
        return None
    
    formatted = []
    for call in tool_calls:
        name = call.get('name', 'unknown')
        args = call.get('arguments', {})
        formatted.append(f"  → {name}({json.dumps(args)})")
    return "\n".join(formatted)


def format_conversation(conversation):
    """Format the conversation in a readable way."""
    output = []
    
    for turn in conversation:
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        tool_calls = turn.get('tool_calls')
        turn_idx = turn.get('turn_idx', '?')
        
        # Format header based on role
        if role == 'assistant':
            header = f"\n{'='*80}\n[Turn {turn_idx}] 🤖 ASSISTANT\n{'='*80}"
        elif role == 'user':
            header = f"\n{'='*80}\n[Turn {turn_idx}] 👤 USER\n{'='*80}"
        elif role == 'tool':
            header = f"\n{'='*80}\n[Turn {turn_idx}] 🔧 TOOL RESULT\n{'='*80}"
        else:
            header = f"\n{'='*80}\n[Turn {turn_idx}] {role.upper()}\n{'='*80}"
        
        output.append(header)
        
        # Display content if present
        if content:
            output.append(content)
        
        # Display tool calls if present
        if tool_calls:
            output.append("\n📞 Tool Calls:")
            output.append(format_tool_calls(tool_calls))
    
    return "\n".join(output)


def format_action_sequence(action_checks, expected_actions=None):
    """Format action sequences for comparison."""
    output = []
    
    # Get model's actual actions
    model_actions = []
    for check in action_checks:
        action = check.get('action', {})
        match = check.get('action_match', False)
        reward = check.get('action_reward', 0)
        model_actions.append({
            'name': action.get('name', 'unknown'),
            'arguments': action.get('arguments', {}),
            'match': match,
            'reward': reward
        })
    
    # Format expected actions if available
    if expected_actions and len(expected_actions) > 0:
        output.append("\n" + "="*80)
        output.append("ACTION SEQUENCE COMPARISON")
        output.append("="*80)
        
        max_len = max(len(model_actions), len(expected_actions))
        
        output.append(f"\n{'Index':<6} {'Expected Action':<40} {'Model Action':<40} {'Match':<8} {'Reward':<8}")
        output.append("-" * 100)
        
        for i in range(max_len):
            expected = expected_actions[i] if i < len(expected_actions) else None
            model = model_actions[i] if i < len(model_actions) else None
            
            expected_str = "N/A"
            if expected:
                name = expected.get('name', 'unknown')
                args = expected.get('arguments', {})
                expected_str = f"{name}({json.dumps(args)})"
            
            model_str = "N/A"
            match_str = "N/A"
            reward_str = "N/A"
            if model:
                name = model['name']
                args = model['arguments']
                model_str = f"{name}({json.dumps(args)})"
                match_str = "✓" if model['match'] else "✗"
                reward_str = str(model['reward'])
            
            output.append(f"{i+1:<6} {expected_str:<40} {model_str:<40} {match_str:<8} {reward_str:<8}")
    else:
        # If no expected actions, just show model's actions
        output.append("\n" + "="*80)
        output.append("MODEL ACTION SEQUENCE")
        output.append("="*80)
        
        for i, model in enumerate(model_actions, 1):
            name = model['name']
            args = model['arguments']
            match_str = "✓" if model['match'] else "✗"
            reward = model['reward']
            output.append(f"\n{i}. {match_str} {name}({json.dumps(args)}) - Reward: {reward}")
    
    return "\n".join(output)


def visualize_conversation(task_id, domain, trial=0):
    """Main function to visualize a conversation."""
    
    # Construct file path
    if domain not in ['retail', 'airline']:
        print(f"Error: domain must be 'retail' or 'airline', got '{domain}'")
        sys.exit(1)
    
    file_path = Path(f"/home/ubuntu/tarun/games/qwen3-4b-instruct-2507-{domain}-detailed-log.jsonl")
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Search for the task
    found = False
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Check if this is the task we're looking for
                if str(data.get('task_id')) == str(task_id) and data.get('trial') == trial:
                    found = True
                    
                    # Print task information
                    print("\n" + "="*80)
                    print("TASK INFORMATION")
                    print("="*80)
                    print(f"Task ID: {data.get('task_id')}")
                    print(f"Trial: {data.get('trial')}")
                    print(f"Domain: {data.get('domain')}")
                    print(f"Success: {data.get('success')}")
                    print(f"Reward: {data.get('reward')}")
                    
                    # Print task goal
                    task = data.get('task', {})
                    goal = task.get('goal', 'N/A')
                    print(f"\nGoal: {goal}")
                    
                    # Print user scenario if available
                    full_task = task.get('full_task', {})
                    user_scenario = full_task.get('user_scenario', {})
                    instructions = user_scenario.get('instructions', {})
                    
                    if instructions:
                        print("\n" + "-"*80)
                        print("USER SCENARIO")
                        print("-"*80)
                        print(f"Reason for call: {instructions.get('reason_for_call', 'N/A')}")
                        print(f"Known info: {instructions.get('known_info', 'N/A')}")
                        print(f"Unknown info: {instructions.get('unknown_info', 'N/A')}")
                    
                    # Print the conversation
                    conversation = data.get('conversation', [])
                    if conversation:
                        print("\n" + "="*80)
                        print("CONVERSATION")
                        print("="*80)
                        print(format_conversation(conversation))
                    else:
                        print("\nNo conversation found.")
                    
                    # Print evaluation summary
                    evaluation = data.get('evaluation', {})
                    if evaluation:
                        print("\n" + "="*80)
                        print("EVALUATION SUMMARY")
                        print("="*80)
                        print(f"DB Check: {evaluation.get('db_check')}")
                        print(f"Communicate Check: {evaluation.get('communicate_check')}")
                        
                        # Get expected actions from ground_truth
                        ground_truth = data.get('ground_truth', {})
                        expected_actions = ground_truth.get('expected_actions', [])
                        
                        action_checks = evaluation.get('action_checks', [])
                        if action_checks:
                            print(format_action_sequence(action_checks, expected_actions))
                            
                            # Summary statistics
                            total_actions = len(action_checks)
                            matched_actions = sum(1 for ac in action_checks if ac.get('action_match', False))
                            failed_actions = total_actions - matched_actions
                            
                            print(f"\n{'='*80}")
                            print("ACTION SUMMARY")
                            print(f"{'='*80}")
                            print(f"Total actions: {total_actions}")
                            print(f"Matched actions: {matched_actions} (✓)")
                            print(f"Failed actions: {failed_actions} (✗)")
                            
                            if expected_actions:
                                print(f"Expected actions: {len(expected_actions)}")
                                if len(expected_actions) != total_actions:
                                    print(f"⚠️  Note: Expected sequence length ({len(expected_actions)}) differs from model sequence length ({total_actions})")
                    
                    break
            
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {e}")
                continue
    
    if not found:
        print(f"Error: Task ID '{task_id}' with trial {trial} not found in {domain} domain.")
        sys.exit(1)


def main():
    """Entry point for the script."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    task_id = sys.argv[1]
    domain = sys.argv[2]
    trial = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    visualize_conversation(task_id, domain, trial)


if __name__ == "__main__":
    main()
