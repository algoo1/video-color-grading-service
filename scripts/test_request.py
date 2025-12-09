import requests
import sys
import os

def test_api(video_path, ref_path=None):
    url = "http://localhost:8000/process"
    
    files = {
        'video_file': open(video_path, 'rb')
    }
    if ref_path:
        files['reference_image'] = open(ref_path, 'rb')
        
    data = {
        'quality_mode': 'fast',
        'stabilization': 'true'
    }
    
    print(f"Sending request to {url}...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        print("Success!")
        print(response.json())
    else:
        print("Failed!")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_request.py <video_path> [ref_image_path]")
        sys.exit(1)
        
    video = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_api(video, ref)
