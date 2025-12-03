import cv2
import numpy as np
import time
import os
from datetime import datetime
import torch
from collections import deque

# Add this line to avoid the OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class StampedeDetectionSystem:
    def __init__(self, video_source=0, 
                 detection_threshold=0.01,
                 crowd_density_threshold=0.4, 
                 movement_threshold=50,
                 save_frames=True, 
                 frame_output_dir="detected_frames/"
                 ):
        """
        Initialize the Stampede Detection System
            video_source: Camera index or video file path
            detection_threshold: Confidence threshold for person detection
            crowd_density_threshold: Threshold for crowd density (0-1)
            movement_threshold: Threshold for movement intensity
            save_frames: Whether to save frames when stampede is detected
            frame_output_dir: Directory to save detected frames
        """
        self.video_source = video_source
        self.detection_threshold = detection_threshold
        self.crowd_density_threshold = crowd_density_threshold
        self.movement_threshold = movement_threshold
        self.save_frames = save_frames
        self.frame_output_dir = frame_output_dir

        # Initialize video capture
        self.cap = None

        # Load YOLOv8 model
        print("Loading YOLOv8 person detection model...")
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8x.pt')

        except Exception as e:
            print(f"Failed to load YOLOv8 model: {e}")
            print("Using fallback detection method.")
            self.model = None

        # Load cascade classifier for head detection
        print("Loading head detection model...")
        try:
            # Load Haar cascade for head detection
            self.head_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Head detection model loaded successfully.")
        except Exception as e:
            print(f"Failed to load head detection model: {e}")
            self.head_cascade = None

        # Initialize tracking
        self.next_track_id = 0
        self.tracks = {}  # Dictionary to store active tracks: {id: {box, age, etc.}}
        self.max_track_age = 30  # Increase max age for better tracking of intermittent detections

        # Initialize variables for motion detection
        self.prev_frame = None
        self.prev_positions = []
        self.alerts = []
        self.alert_status = False
        self.alert_start_time = None

        # Motion history for improved detection
        self.motion_history = deque(maxlen=20)

        # Create output directory if saving frames
        if self.save_frames:
            os.makedirs(self.frame_output_dir, exist_ok=True)

    def start_video_capture(self):
        """Initialize the video capture from the source"""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise Exception(f"Error: Could not open video source {self.video_source}")
        return self.cap.isOpened()

    def stop_video_capture(self):
        """Release the video capture resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def convert_video_to_frames(self, max_frames=None):
        """
        Convert video to individual frames for analysis
            max_frames: Maximum number of frames to process (None for all)
        Returns:Generator yielding (frame_number, frame) tuples
        """
        if not self.cap or not self.cap.isOpened():
            self.start_video_capture()

        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            yield frame_count, frame

            if max_frames and frame_count >= max_frames:
                break

    def detect_people_and_heads(self, frame):
        """
        Enhanced detection that combines full-body detection with head detection
        for better accuracy with distant or partially visible people

            frame: Input image frame

        Returns:
            List of bounding boxes for detected people [(x, y, w, h, confidence), ...]
        """
        person_boxes = []

        # Method 1: YOLOv8 for full body detection
        if self.model is not None:
            # Increase size parameter for better small object detection
            results = self.model(frame, classes=[0], conf=self.detection_threshold)  # Only detect people

            # Parse results
            for result in results:
                # Extract boxes, confidence scores
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    person_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(conf)))

        # Method 2: Enhance with head detection using Haar cascade
        if self.head_cascade is not None:
            # Convert frame to grayscale for Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect heads
            heads = self.head_cascade.detectMultiScale(gray, 1.1, 5, minSize=(10, 10))

            # For each detected head, create a person box (estimate full body based on head position)
            for (hx, hy, hw, hh) in heads:
                # Estimate person box based on head position
                # Head is typically in the top 1/6 of a standing person
                person_height = int(hh * 6)  # Estimate full height as 6x head height
                person_width = int(hw * 1.5)  # Estimate width as 1.5x head width

                # Adjust person box to start from head position
                px = max(0, int(hx - person_width * 0.25))  # Center the head in the body box
                py = max(0, hy)  # Keep top of head at top of box
                pw = min(frame.shape[1] - px, person_width)
                ph = min(frame.shape[0] - py, person_height)

                # Add to person boxes with a reasonable confidence
                # Only add if not already detected by YOLO to avoid duplicates
                is_duplicate = False
                for (x, y, w, h, _) in person_boxes:
                    # Check if head center is inside an existing person box
                    head_center_x = hx + hw // 2
                    head_center_y = hy + hh // 2
                    if (x <= head_center_x <= x + w) and (y <= head_center_y <= y + h):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    person_boxes.append((px, py, pw, ph, 0.6))  # Fixed confidence for head detections

        # Method 3: Add HOG detector as fallback
        if len(person_boxes) == 0:
            # Use HOG detector as fallback when no detections from other methods
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            # Adjust parameters for better detection of distant people
            hog_boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(2, 2), scale=1.02)

            for (x, y, w, h), conf in zip(hog_boxes, weights):
                person_boxes.append((int(x), int(y), int(w), int(h), float(conf)))

        return person_boxes

    def track_people(self, person_boxes, frame_width, frame_height):
        """
        Track people across frames for more stable counting

        Args:
            person_boxes: List of bounding boxes with confidence [(x, y, w, h, conf), ...]
            frame_width: Width of the frame
            frame_height: Height of the frame

        Returns:
            List of tracked bounding boxes with IDs [(x, y, w, h, track_id), ...]
        """
        # Update age of all existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            # Remove old tracks
            if self.tracks[track_id]['age'] > self.max_track_age:
                del self.tracks[track_id]

        # Match new detections to existing tracks
        matched_track_ids = []
        result_boxes = []

        for x, y, w, h, conf in person_boxes:
            center_x = x + w // 2
            center_y = y + h // 2

            best_track_id = None
            best_distance = float('inf')

            # Find the closest track
            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue

                track_center_x = track['box'][0] + track['box'][2] // 2
                track_center_y = track['box'][1] + track['box'][3] // 2

                # Calculate Euclidean distance
                distance = np.sqrt((center_x - track_center_x) ** 2 + (center_y - track_center_y) ** 2)

                # Use distance threshold proportional to frame size
                distance_threshold = (frame_width + frame_height) * 0.15  # Increased from 0.1 to 0.15

                if distance < distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['box'] = (x, y, w, h)
                self.tracks[best_track_id]['age'] = 0
                self.tracks[best_track_id]['conf'] = conf
                matched_track_ids.append(best_track_id)
                result_boxes.append((x, y, w, h, best_track_id))
            else:
                # Create new track
                new_track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[new_track_id] = {
                    'box': (x, y, w, h),
                    'age': 0,
                    'conf': conf
                }
                result_boxes.append((x, y, w, h, new_track_id))

        # Include tracks that weren't matched but are still relatively fresh
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids and track[
                'age'] < 5:  # Only include tracks that are less than 5 frames old
                x, y, w, h = track['box']
                result_boxes.append((x, y, w, h, track_id))

        return result_boxes

    def analyze_crowd_density(self, frame, tracked_boxes):
        """
        Analyze crowd density based on detected people

        Args:
            frame: Input image frame
            tracked_boxes: List of tracked bounding boxes [(x, y, w, h, track_id), ...]

        Returns:
            density_score: Value between 0-1 indicating crowd density
            density_frame: Frame with density visualization
        """
        height, width = frame.shape[:2]
        frame_area = height * width

        # Calculate area covered by people
        people_area = 0
        for (x, y, w, h, _) in tracked_boxes:
            people_area += w * h

        # Calculate density as ratio of area covered by people to total frame area
        density_score = min(1.0, people_area / frame_area)

        # Apply non-linear scaling to make the density score more sensitive
        # This helps to detect stampedes even when people are far away
        adjusted_density = min(1.0, density_score * 1.5)

        # Create a visualization
        density_frame = frame.copy()

        # Draw people count prominently
        people_count = len(tracked_boxes)
        cv2.putText(density_frame, f"PEOPLE COUNT: {people_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(density_frame, f"Density: {adjusted_density:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw boxes around tracked people with ID numbers
        for i, (x, y, w, h, track_id) in enumerate(tracked_boxes):
            # Use a thicker, more visible bounding box
            cv2.rectangle(density_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Create a filled rectangle for better text visibility
            label_bg_color = (0, 200, 0)
            label_text_color = (0, 0, 0)
            label_text = f"ID: {track_id}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Draw background for label
            cv2.rectangle(density_frame,
                          (x, y - 20),
                          (x + label_size[0] + 10, y),
                          label_bg_color, -1)

            # Draw label text
            cv2.putText(density_frame, label_text, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

        return adjusted_density, density_frame

    def detect_motion(self, frame, tracked_boxes):
        """
        Detect anomalous motion patterns that might indicate stampede

        Args:
            frame: Input image frame
            tracked_boxes: List of tracked bounding boxes

        Returns:
            motion_score: Value indicating motion intensity
            motion_frame: Frame with motion visualization
        """
        # Convert frame to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize previous frame if needed
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0, frame.copy()

        # Calculate absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Calculate motion score as percentage of pixels with motion
        motion_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_score = motion_pixels / total_pixels * 100

        # Add motion score to history for smoothing
        self.motion_history.append(motion_score)
        avg_motion = sum(self.motion_history) / len(self.motion_history)

        # Create a visualization
        motion_frame = frame.copy()
        cv2.putText(motion_frame, f"Motion: {avg_motion:.2f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Overlay motion visualization
        motion_overlay = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        motion_overlay[:, :, 0] = 0  # Set blue channel to 0
        motion_overlay[:, :, 1] = 0  # Set green channel to 0
        motion_frame = cv2.addWeighted(motion_frame, 1, motion_overlay, 0.3, 0)

        # Update previous frame
        self.prev_frame = gray

        return avg_motion, motion_frame

    def check_stampede_conditions(self, density_score, motion_score, frame):
        """
        Check if current conditions indicate a possible stampede

        Args:
            density_score: Crowd density score
            motion_score: Motion intensity score
            frame: Current frame for saving if needed

        Returns:
            is_stampede: Boolean indicating if stampede is detected
            alert_message: Alert message if stampede is detected
        """
        # Enhanced stampede detection logic that considers both density and motion
        # with weighted importance

        # For small crowds (low density), require higher motion to trigger alert
        if density_score < self.crowd_density_threshold:
            weighted_score = (density_score / self.crowd_density_threshold * 0.7) + (
                        motion_score / self.movement_threshold * 0.3)
            is_stampede = weighted_score > 1.0 and motion_score > self.movement_threshold * 1.5
        else:
            # For dense crowds, be more sensitive to motion
            weighted_score = (density_score / self.crowd_density_threshold * 0.5) + (
                        motion_score / self.movement_threshold * 0.5)
            is_stampede = weighted_score > 1.0

        # Generate alert message
        alert_message = None
        if is_stampede:
            # Start alert if not already in alert state
            if not self.alert_status:
                self.alert_status = True
                self.alert_start_time = datetime.now()
                alert_message = f" POTENTIAL STAMPEDE DETECTED at {self.alert_start_time.strftime('%H:%M:%S')}"
                self.alerts.append(alert_message)

                # Save frame if enabled
                if self.save_frames:
                    filename = f"{self.frame_output_dir}/stampede_{self.alert_start_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, frame)
        else:
            # End alert if previously in alert state
            if self.alert_status:
                self.alert_status = False
                end_time = datetime.now()
                duration = (end_time - self.alert_start_time).total_seconds()
                alert_message = f"Alert ended at {end_time.strftime('%H:%M:%S')}. Duration: {duration:.1f} seconds"
                self.alerts.append(alert_message)

        return is_stampede, alert_message

    def process_frame(self, frame):
        """
        Process a single frame for stampede detection

        Args:
            frame: Input image frame

        Returns:
            result_frame: Frame with visualizations
            is_stampede: Boolean indicating if stampede is detected
            alert_message: Alert message if stampede is detected
        """
        # Get frame dimensions
        height, width = frame.shape[:2]

        # Detect people and heads in the frame
        person_boxes = self.detect_people_and_heads(frame)

        # Track detections across frames
        tracked_boxes = self.track_people(person_boxes, width, height)

        # Create a copy of the original frame for visualization
        result_frame = frame.copy()

        # Draw bounding boxes for tracked people
        for x, y, w, h, track_id in tracked_boxes:
            # Draw a thick rectangle around each person
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Create better visible label
            label_text = f"ID: {track_id}"
            # Draw background for label
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame,
                          (x, y - 25),
                          (x + label_size[0] + 10, y),
                          (0, 200, 0), -1)
            # Draw label text
            cv2.putText(result_frame, label_text, (x + 5, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Analyze crowd density
        density_score, _ = self.analyze_crowd_density(frame, tracked_boxes)

        # Detect motion
        motion_score, _ = self.detect_motion(frame, tracked_boxes)

        # Check for stampede conditions
        is_stampede, alert_message = self.check_stampede_conditions(
            density_score, motion_score, frame)

        # Display density and motion information
        cv2.putText(result_frame, f"Density: {density_score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Motion: {motion_score:.2f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Add alert status and person count
        alert_color = (0, 0, 255) if is_stampede else (0, 255, 0)
        status_text = "STAMPEDE ALERT" if is_stampede else "Normal"
        cv2.putText(result_frame, status_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)

        # Add counter in top-right corner for better visibility
        h, w = result_frame.shape[:2]
        people_count = len(tracked_boxes)
        count_text = f"PEOPLE: {people_count}"
        text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]

        # Draw background rectangle for counter
        cv2.rectangle(result_frame,
                      (w - text_size[0] - 20, 10),
                      (w - 10, 50),
                      (0, 0, 0), -1)

        # Draw counter text
        cv2.putText(result_frame, count_text,
                    (w - text_size[0] - 15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Draw border around frame when in alert state
        if is_stampede:
            h, w = result_frame.shape[:2]
            cv2.rectangle(result_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

        return result_frame, is_stampede, alert_message

    def run(self, display=True, output_file=None):
        """
        Run the stampede detection system on the video source

        Args:
            display: Whether to display the video output
            output_file: Path to save output video (None to disable)
        """
        if not self.start_video_capture():
            return

        # Get video properties for output video
        if output_file:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # Process each frame
        for frame_count, frame in self.convert_video_to_frames():
            result_frame, is_stampede, alert_message = self.process_frame(frame)

            # Print alert message if any
            if alert_message:
                print(alert_message)

            # Save to output video if enabled
            if output_file:
                out.write(result_frame)

            # Display the frame if enabled
            if display:
                cv2.imshow('Stampede Detection System', result_frame)

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Clean up
        self.stop_video_capture()
        if output_file:
            out.release()

        # Print summary of alerts
        print(f"\nDetection complete. {len(self.alerts)} alerts generated.")
        for alert in self.alerts:
            print(alert)