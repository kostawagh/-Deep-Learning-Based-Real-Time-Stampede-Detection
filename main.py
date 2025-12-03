import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import stamp as st  # Import the module


def main():
    """
    Example usage of the updated Stampede Detection System
    """
    # Use crowd10.mp4 as the video source
    video_source = "crowd9.mp4"

    # Check if file exists
    if not os.path.exists(video_source):
        print(f"Error: Could not find {video_source}. Please make sure the file exists in the current directory.")
        return

    print(f"Starting Stampede Detection System with video file: {video_source}")

    '''
    detection_threshold: Confidence threshold for person detection
    crowd_density_threshold: Threshold for crowd density (0-1)
    movement_threshold: Threshold for movement intensity
    save_frames: Whether to save frames when stampede is detected
    frame_output_dir: Directory to save detected frames
    '''

    # Create the detection system with optimized parameters
    system = st.StampedeDetectionSystem(
        video_source=video_source,
        detection_threshold=0.01,  # Lowered threshold for YOLOv8 to detect people
        crowd_density_threshold=0.7,  # threshold for stampede detection
        movement_threshold= 3,  # Lower threshold for more sensitive motion detection
        save_frames=True
    )

    # Run the system
    system.run(display=True, output_file="stampede_detection_output.avi")


if __name__ == "__main__":
    main()