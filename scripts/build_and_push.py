import os
import sys
import subprocess
import shutil

def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        sys.exit(1)

def main():
    print("==================================================")
    print("  VideoColorGrading Build & Push Script (Python)")
    print("==================================================")

    if len(sys.argv) < 2:
        print("Error: Docker Hub username is required.")
        print("Usage: python build_and_push.py <your_dockerhub_username>")
        sys.exit(1)

    docker_username = sys.argv[1]
    image_name = "video-grading"
    tag = "v1"
    full_image_name = f"{docker_username}/{image_name}:{tag}"

    print(f"  Target Image: {full_image_name}")
    print("==================================================")

    # Check for weights
    if not os.path.exists("pretrained/gs_extractor.pth") or not os.path.exists("pretrained/l_diffuser.pth"):
        print("WARNING: Model weights not found in pretrained/ directory.")
        print("Please ensure 'gs_extractor.pth' and 'l_diffuser.pth' are in 'pretrained/' folder.")
        response = input("Do you want to continue anyway (weights will be missing)? [y/N] ")
        if response.lower() != 'y':
            sys.exit(1)

    # Docker Login
    print("\nLogging in to Docker Hub...")
    run_command("docker login")

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
