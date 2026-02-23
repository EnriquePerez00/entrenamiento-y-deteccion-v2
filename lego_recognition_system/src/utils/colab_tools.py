
import subprocess
import shlex
import sys
import os

def run_command(command, cwd=None, env=None):
    """
    Executes a shell command with real-time output streaming to the Colab/Jupyter console.
    
    Args:
        command (str or list): The command to execute.
        cwd (str, optional): Working directory.
        env (dict, optional): Environment variables.
        
    Returns:
        int: The return code of the command.
    """
    if isinstance(command, str):
        cmd_args = shlex.split(command)
    else:
        cmd_args = command

    print(f"Executing: {' '.join(cmd_args)}")

    try:
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1  # Line buffered
        )

        # Stream output line by line
        for line in process.stdout:
            print(line, end="")

        process.wait()
        return process.returncode

    except KeyboardInterrupt:
        print("\n\n[!] Process interrupted by user. Terminating subprocess...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[!] Subprocess did not terminate, killing...")
            process.kill()
        return -1
    except Exception as e:
        print(f"\n[!] Error executing command: {e}")
        return 1
