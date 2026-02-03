#!/usr/bin/env python3
"""
Deterministic vLLM Server Manager

Launches N vLLM server processes on a single GPU, each with --max-num-seqs 1
for deterministic evaluation. Each server gets 0.8/N GPU memory utilization.

Usage:
    # Start servers
    python deterministic_vllm.py start --gpu 4 --num-servers 2 --model Qwen/Qwen3-4B-Instruct-2507 --base-port 9000

    # Stop servers
    python deterministic_vllm.py stop --base-port 9000 --num-servers 2

    # Check status
    python deterministic_vllm.py status --base-port 9000 --num-servers 2
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional

import psutil
import requests


def get_server_ports(base_port: int, num_servers: int) -> List[int]:
    """Get list of ports for N servers starting from base_port."""
    return [base_port + i for i in range(num_servers)]


def kill_process_on_port(port: int) -> bool:
    """Kill any process running on the specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                connections = proc.net_connections()
                for conn in connections:
                    if hasattr(conn, "laddr") and conn.laddr.port == port:
                        print(f"Killing existing process {proc.pid} on port {port}")
                        proc.terminate()
                        time.sleep(0.5)
                        if proc.is_running():
                            proc.kill()
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
    return False


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready on the given port."""
    url = f"http://localhost:{port}/v1/models"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    return False


def start_servers(
    gpu_id: int,
    num_servers: int,
    model: str,
    base_port: int,
    max_model_len: int = 55000,
    total_gpu_memory_util: float = 0.8,
    enable_lora: bool = True,
    max_loras: int = 2,
    extra_args: Optional[List[str]] = None,
) -> List[subprocess.Popen]:
    """
    Start N vLLM servers on a single GPU for deterministic evaluation.

    Args:
        gpu_id: GPU device ID to use
        num_servers: Number of server processes to start
        model: Model name/path to serve
        base_port: Base port number (servers will use base_port, base_port+1, ...)
        max_model_len: Maximum model context length
        total_gpu_memory_util: Total GPU memory utilization to split among servers
        enable_lora: Whether to enable LoRA support
        max_loras: Maximum number of LoRA adapters
        extra_args: Additional arguments to pass to vLLM

    Returns:
        List of Popen process handles
    """
    ports = get_server_ports(base_port, num_servers)
    per_server_memory = total_gpu_memory_util / num_servers

    processes = []
    log_dir = f"/tmp/vllm_logs_gpu{gpu_id}"
    os.makedirs(log_dir, exist_ok=True)

    for i, port in enumerate(ports):
        # Kill any existing process on this port
        kill_process_on_port(port)

        cmd = [
            "vllm", "serve", model,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--dtype", "bfloat16",
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(per_server_memory),
            "--max-num-seqs", "1",  # Critical for determinism
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]

        if enable_lora:
            cmd.extend(["--enable-lora", "--max-loras", str(max_loras)])

        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_file = open(f"{log_dir}/server_{port}.log", "w")

        print(f"Starting vLLM server {i+1}/{num_servers} on port {port} (GPU {gpu_id}, memory: {per_server_memory:.2f})")
        print(f"  Command: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append(proc)

        # Small delay between starting servers
        time.sleep(2)

    return processes


def wait_for_all_servers(ports: List[int], timeout: int = 300) -> bool:
    """Wait for all servers to be ready."""
    print(f"Waiting for {len(ports)} servers to be ready (timeout: {timeout}s)...")

    for port in ports:
        print(f"  Waiting for server on port {port}...", end=" ", flush=True)
        if wait_for_server(port, timeout):
            print("Ready!")
        else:
            print("FAILED!")
            return False

    return True


def stop_servers(base_port: int, num_servers: int) -> None:
    """Stop all servers started from base_port."""
    ports = get_server_ports(base_port, num_servers)

    for port in ports:
        if kill_process_on_port(port):
            print(f"Stopped server on port {port}")
        else:
            print(f"No server found on port {port}")


def check_status(base_port: int, num_servers: int) -> dict:
    """Check status of all servers."""
    ports = get_server_ports(base_port, num_servers)
    status = {}

    for port in ports:
        url = f"http://localhost:{port}/v1/models"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                model_ids = [m["id"] for m in models]
                status[port] = {"status": "running", "models": model_ids}
            else:
                status[port] = {"status": "error", "code": response.status_code}
        except requests.exceptions.RequestException as e:
            status[port] = {"status": "down", "error": str(e)}

    return status


def get_base_urls(base_port: int, num_servers: int) -> List[str]:
    """Get list of base URLs for all servers."""
    ports = get_server_ports(base_port, num_servers)
    return [f"http://localhost:{port}/v1" for port in ports]


def main():
    parser = argparse.ArgumentParser(description="Deterministic vLLM Server Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start vLLM servers")
    start_parser.add_argument("--gpu", type=int, required=True, help="GPU device ID")
    start_parser.add_argument("--num-servers", "-n", type=int, required=True, help="Number of servers")
    start_parser.add_argument("--model", type=str, required=True, help="Model to serve")
    start_parser.add_argument("--base-port", type=int, default=9000, help="Base port number")
    start_parser.add_argument("--max-model-len", type=int, default=55000, help="Max model length")
    start_parser.add_argument("--gpu-memory-util", type=float, default=0.8, help="Total GPU memory utilization")
    start_parser.add_argument("--no-lora", action="store_true", help="Disable LoRA support")
    start_parser.add_argument("--wait", action="store_true", help="Wait for servers to be ready")
    start_parser.add_argument("--timeout", type=int, default=300, help="Timeout for waiting")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop vLLM servers")
    stop_parser.add_argument("--base-port", type=int, default=9000, help="Base port number")
    stop_parser.add_argument("--num-servers", "-n", type=int, required=True, help="Number of servers")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument("--base-port", type=int, default=9000, help="Base port number")
    status_parser.add_argument("--num-servers", "-n", type=int, required=True, help="Number of servers")

    # URLs command
    urls_parser = subparsers.add_parser("urls", help="Print server URLs")
    urls_parser.add_argument("--base-port", type=int, default=9000, help="Base port number")
    urls_parser.add_argument("--num-servers", "-n", type=int, required=True, help="Number of servers")

    args = parser.parse_args()

    if args.command == "start":
        processes = start_servers(
            gpu_id=args.gpu,
            num_servers=args.num_servers,
            model=args.model,
            base_port=args.base_port,
            max_model_len=args.max_model_len,
            total_gpu_memory_util=args.gpu_memory_util,
            enable_lora=not args.no_lora,
        )

        if args.wait:
            ports = get_server_ports(args.base_port, args.num_servers)
            if wait_for_all_servers(ports, args.timeout):
                print("\nAll servers are ready!")
                print("Base URLs:")
                for url in get_base_urls(args.base_port, args.num_servers):
                    print(f"  {url}")
            else:
                print("\nSome servers failed to start. Check logs in /tmp/vllm_logs_gpu{args.gpu}/")
                sys.exit(1)
        else:
            print(f"\nServers starting in background. Use 'status' command to check.")
            print(f"Logs: /tmp/vllm_logs_gpu{args.gpu}/")

    elif args.command == "stop":
        stop_servers(args.base_port, args.num_servers)

    elif args.command == "status":
        status = check_status(args.base_port, args.num_servers)
        for port, info in status.items():
            print(f"Port {port}: {info}")

    elif args.command == "urls":
        for url in get_base_urls(args.base_port, args.num_servers):
            print(url)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
