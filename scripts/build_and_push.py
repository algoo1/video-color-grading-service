import os
import sys
import subprocess
import shutil
import getpass

def run_command(command, stdin=None):
    try:
        if stdin:
            subprocess.check_call(command, shell=True, input=stdin, text=True)
        else:
            subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        sys.exit(1)

def load_env_file():
    """Simple parser for .env files to avoid external dependencies"""
    env_vars = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip().strip("'").strip('"')
    return env_vars

def main():
    print("==================================================")
    print("  VideoColorGrading Build & Push Script (Python)")
    print("==================================================")

    # Load from .env
    env_vars = load_env_file()
    
    # Prioritize Env Vars > .env file > User Input
    docker_username = os.environ.get("DOCKER_USERNAME") or env_vars.get("DOCKER_USERNAME")
    docker_password = os.environ.get("DOCKER_PASSWORD") or env_vars.get("DOCKER_PASSWORD")

    # If argument provided, override username (backward compatibility)
    if len(sys.argv) > 1:
        docker_username = sys.argv[1]

    # Prompt if missing
    if not docker_username:
        docker_username = input("Enter Docker Hub Username: ").strip()
    
    if not docker_password:
        print(f"Docker Password for user '{docker_username}' (hidden): ", end="", flush=True)
        docker_password = getpass.getpass(prompt="")

    if not docker_username or not docker_password:
        print("Error: Docker credentials are required.")
        sys.exit(1)

    image_name = "video-grading"
    tag = "v1"
    full_image_name = f"{docker_username}/{image_name}:{tag}"

    print(f"\n  Target Image: {full_image_name}")
    print("==================================================")

    # Check for weights
    if not os.path.exists("pretrained/gs_extractor.pth") or not os.path.exists("pretrained/l_diffuser.pth"):
        print("WARNING: Model weights not found in pretrained/ directory.")
        print("Please ensure 'gs_extractor.pth' and 'l_diffuser.pth' are in 'pretrained/' folder.")
        response = input("Do you want to continue anyway (weights will be missing)? [y/N] ")
        if response.lower() != 'y':
            sys.exit(1)

    # Docker Login (Securely via stdin)
    print("\nLogging in to Docker Hub...")
    # Passing password via stdin prevents it from showing up in process list/history
    run_command(f"docker login -u {docker_username} --password-stdin", stdin=docker_password)

    # Build
    print(f"\nBuilding Docker Image: {full_image_name}...")
    run_command(f"docker build -t {full_image_name} .")

    # Push
    print(f"\nPushing to Docker Hub: {full_image_name}...")
    run_command(f"docker push {full_image_name}")

    print("\n==================================================")
    print("  SUCCESS!")
    print(f"  Image pushed to: {full_image_name}")
    print("==================================================")
    print("  Now go to RunPod Serverless -> New Endpoint")
    print(f"  Container Image: {full_image_name}")
    print("  Docker Command: python runpod_handler.py")
    print("==================================================")


if __name__ == "__main__":
    main()
