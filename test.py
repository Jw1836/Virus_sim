from moviepy.editor import VideoFileClip, concatenate_videoclips

gif_path = "/Users/jwayment/Code/Virus_sim/phase_plot.gif"
output_video_path = "my_animation_video.mp4"

try:
    # Load the GIF as a video clip
    clip = VideoFileClip(gif_path)

    # Add a 2-second pause at the start (freeze first frame)
    pause = clip.to_ImageClip(duration=2)

    # Speed up the animation (e.g., 2x faster)
    fast_clip = clip.fx(vfx.speedx, 2)

    # Concatenate pause and fast-forwarded animation
    final_clip = concatenate_videoclips([pause, fast_clip])

    # Write the clip to a video file (e.g., MP4)
    final_clip.write_videofile(output_video_path)

    print(f"Successfully converted '{gif_path}' to '{output_video_path}' with pause and fast-forward.")

except FileNotFoundError:
    print(f"Error: GIF file not found at '{gif_path}'")
except Exception as e:
    print(f"An error occurred: {e}")