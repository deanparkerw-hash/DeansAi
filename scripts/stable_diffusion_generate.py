import os
from stable_diffusion_model import generate_frame  # Assuming a function to generate individual frames

def generate_frames(prompt, style):
    os.makedirs('frames', exist_ok=True)  # Create a directory for frames if it doesn't exist

    for i in range(1, 13):
        frame_filename = f'frames/frame_{i:03}.png'
        generate_frame(prompt, style, frame_filename)  # Generate the frame

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    style = input("Enter your style: ")
    generate_frames(prompt, style)