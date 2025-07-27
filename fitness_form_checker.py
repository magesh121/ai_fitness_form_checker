# Heat map visualization for muscle stress (enhanced version)
def create_detailed_stress_heatmap(features, prediction, stress_data):
    """Create a detailed heatmap showing areas of concern with muscle groups"""
    if features is None or not stress_data:
        return None
    
    # Create a figure for the heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Stress levels by body part
    body_parts = [part.replace('_', ' ').title() for part in stress_data.keys()]
    stress_levels = list(stress_data.values())
    
    # Color map for stress levels
    colors = ['green' if s < 0.3 else 'yellow' if s < 0.7 else 'red' for s in stress_levels]
    
    # Bar chart of stress levels
    bars = ax1.barh(body_parts, stress_levels, color=colors, alpha=0.7)
    ax1.set_xlabel('Stress Level')
    ax1.set_title('Body Part Stress Analysis')
    ax1.set_xlim(0, 1)
    
    # Add stress threshold lines
    ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='Low-Medium')
    ax1.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Medium-High')
    ax1.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars, stress_levels):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}', va='center', fontsize=9)
    
    # Body diagram representation
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 15)
    ax2.set_aspect('equal')
    ax2.set_title('Stress Distribution Map')
    
    # Simple body outline
    # Head
    head = plt.Circle((5, 13), 0.8, fill=False, color='black', linewidth=2)
    ax2.add_patch(head)
    
    # Torso
    spine_stress = stress_data.get('spine_stress', 0)
    spine_color = 'red' if spine_stress > 0.7 else 'yellow' if spine_stress > 0.3 else 'green'
    ax2.plot([5, 5], [12, 8], color=spine_color, linewidth=6, alpha=0.8, label='Spine')
    
    # Arms (simplified)
    ax2.plot([5, 3], [11, 9], 'gray', linewidth=2, alpha=0.5)
    ax2.plot([5, 7], [11, 9], 'gray', linewidth=2, alpha=0.5)
    
    # Left leg with stress visualization
    left_knee_stress = stress_data.get('left_knee_stress', 0)
    left_hip_stress = stress_data.get('left_hip_stress', 0)
    left_ankle_stress = stress_data.get('left_ankle_stress', 0)
    
    # Thigh
    thigh_color = 'red' if left_knee_stress > 0.7 else 'yellow' if left_knee_stress > 0.3 else 'green'
    ax2.plot([5, 4], [8, 4], color=thigh_color, linewidth=8, alpha=0.8)
    
    # Calf
    calf_color = 'red' if left_ankle_stress > 0.7 else 'yellow' if left_ankle_stress > 0.3 else 'green'
    ax2.plot([4, 4], [4, 1], color=calf_color, linewidth=8, alpha=0.8)
    
    # Right leg with stress visualization
    right_knee_stress = stress_data.get('right_knee_stress', 0)
    right_hip_stress = stress_data.get('right_hip_stress', 0)
    right_ankle_stress = stress_data.get('right_ankle_stress', 0)
    
    # Thigh
    thigh_color = 'red' if right_knee_stress > 0.7 else 'yellow' if right_knee_stress > 0.3 else 'green'
    ax2.plot([5, 6], [8, 4], color=thigh_color, linewidth=8, alpha=0.8)
    
    # Calf
    calf_color = 'red' if right_ankle_stress > 0.7 else 'yellow' if right_ankle_stress > 0.3 else 'green'
    ax2.plot([6, 6], [4, 1], color=calf_color, linewidth=8, alpha=0.8)
    
    # Joint markers
    joints = [
        (5, 8, max(left_hip_stress, right_hip_stress), 'Hip'),
        (4, 4, left_knee_stress, 'L.Knee'),
        (6, 4, right_knee_stress, 'R.Knee'),
        (4, 1, left_ankle_stress, 'L.Ankle'),
        (6, 1, right_ankle_stress, 'R.Ankle')
    ]
    
    for x, y, stress, label in joints:
        joint_color = 'red' if stress > 0.7 else 'yellow' if stress > 0.3 else 'green'
        circle = plt.Circle((x, y), 0.3, color=joint_color, alpha=0.8)
        ax2.add_patch(circle)
        ax2.text(x + 0.5, y, f'{label}\n{stress:.2f}', fontsize=8, ha='left')
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=4, alpha=0.8, label='Low Stress (<0.3)'),
        plt.Line2D([0], [0], color='yellow', lw=4, alpha=0.8, label='Medium Stress (0.3-0.7)'),
        plt.Line2D([0], [0], color='red', lw=4, alpha=0.8, label='High Stress (>0.7)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_real_time_heatmap_display(stress_data):
    """Create a real-time heatmap display for Streamlit"""
    if not stress_data:
        return None
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    body_parts = [part.replace('_', ' ').title() for part in stress_data.keys()]
    stress_levels = list(stress_data.values())
    
    # Create horizontal bar chart
    colors = []
    for stress in stress_levels:
        if stress < 0.3:
            colors.append('#00FF00')  # Green
        elif stress < 0.7:
            colors.append('#FFFF00')  # Yellow
        else:
            colors.append('#FF0000')  # Red
    
    bars = ax.barh(body_parts, stress_levels, color=colors, alpha=0.7)
    
    # Customize chart
    ax.set_xlabel('Stress Level', fontsize=12)
    ax.set_title('Real-Time Body Stress Analysis', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add threshold lines
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add value labels
    for bar, value in zip(bars, stress_levels):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}', va='center', fontweight='bold')
    
    # Grid and styling
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00FF00', alpha=0.7, label='Low Stress (< 0.3)'),
        Patch(facecolor='#FFFF00', alpha=0.7, label='Medium Stress (0.3 - 0.7)'),
        Patch(facecolor='#FF0000', alpha=0.7, label='High Stress (> 0.7)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return fig# Real-Time Fitness Form Checker
# A complete ML/DL system for detecting exercise form using MediaPipe and OpenCV

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import math
import warnings
warnings.filterwarnings('ignore')

class PoseExtractor:
    """Class to handle MediaPipe pose detection and feature extraction"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_landmarks(self, image):
        """Extract pose landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        return results
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def extract_squat_features(self, landmarks):
        """Extract features specific to squat exercise"""
        if not landmarks.pose_landmarks:
            return None
        
        # Get key landmarks for squat analysis
        points = landmarks.pose_landmarks.landmark
        
        # Hip, knee, ankle points for both legs
        left_hip = [points[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   points[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [points[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    points[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        right_hip = [points[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    points[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Shoulder and spine points
        left_shoulder = [points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        
        # Calculate angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # Hip angles (using shoulder-hip-knee)
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Spine angle (shoulder to hip vertical alignment)
        spine_angle = abs(left_shoulder[0] - left_hip[0])  # Horizontal deviation
        
        # Knee alignment (knees should track over toes)
        left_knee_alignment = abs(left_knee[0] - left_ankle[0])
        right_knee_alignment = abs(right_knee[0] - right_ankle[0])
        
        # Hip depth (how low the person goes)
        hip_depth = min(left_hip[1], right_hip[1])
        knee_level = min(left_knee[1], right_knee[1])
        squat_depth = hip_depth - knee_level
        
        features = [
            left_knee_angle, right_knee_angle,
            left_hip_angle, right_hip_angle,
            spine_angle,
            left_knee_alignment, right_knee_alignment,
            squat_depth
        ]
        
        return features
    
    def draw_heatmap_overlay(self, image, landmarks, stress_data, prediction):
        """Draw heatmap overlay on the user's body based on form analysis"""
        if not landmarks.pose_landmarks:
            return image
            
        h, w, _ = image.shape
        overlay = image.copy()
        
        # Define body segments and their stress levels
        segments = self.get_body_segments(landmarks, w, h, stress_data)
        
        # Draw heatmap zones
        for segment_name, segment_data in segments.items():
            if segment_data['stress'] > 0:
                self.draw_stress_zone(overlay, segment_data, segment_name)
        
        # Draw enhanced pose landmarks with color coding
        self.draw_enhanced_landmarks(overlay, landmarks, stress_data, w, h)
        
        # Add form quality indicators
        self.draw_form_indicators(overlay, prediction, w, h)
        
        # Blend overlay with original image
        alpha = 0.6
        result = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
        
        return result
    
    def get_body_segments(self, landmarks, width, height, stress_data):
        """Define body segments for heatmap visualization"""
        points = landmarks.pose_landmarks.landmark
        
        # Convert normalized coordinates to pixel coordinates
        def get_pixel_coords(landmark_idx):
            return (int(points[landmark_idx].x * width), 
                   int(points[landmark_idx].y * height))
        
        # Key body points
        left_shoulder = get_pixel_coords(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_pixel_coords(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_hip = get_pixel_coords(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = get_pixel_coords(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        left_knee = get_pixel_coords(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        right_knee = get_pixel_coords(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        left_ankle = get_pixel_coords(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        right_ankle = get_pixel_coords(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        
        segments = {
            'left_thigh': {
                'points': [left_hip, left_knee],
                'stress': stress_data.get('left_knee_stress', 0),
                'center': ((left_hip[0] + left_knee[0]) // 2, (left_hip[1] + left_knee[1]) // 2)
            },
            'right_thigh': {
                'points': [right_hip, right_knee],
                'stress': stress_data.get('right_knee_stress', 0),
                'center': ((right_hip[0] + right_knee[0]) // 2, (right_hip[1] + right_knee[1]) // 2)
            },
            'left_calf': {
                'points': [left_knee, left_ankle],
                'stress': stress_data.get('left_ankle_stress', 0),
                'center': ((left_knee[0] + left_ankle[0]) // 2, (left_knee[1] + left_ankle[1]) // 2)
            },
            'right_calf': {
                'points': [right_knee, right_ankle],
                'stress': stress_data.get('right_ankle_stress', 0),
                'center': ((right_knee[0] + right_ankle[0]) // 2, (right_knee[1] + right_ankle[1]) // 2)
            },
            'spine': {
                'points': [((left_shoulder[0] + right_shoulder[0]) // 2, 
                           (left_shoulder[1] + right_shoulder[1]) // 2),
                          ((left_hip[0] + right_hip[0]) // 2, 
                           (left_hip[1] + right_hip[1]) // 2)],
                'stress': stress_data.get('spine_stress', 0),
                'center': (((left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) // 4,
                           (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) // 4))
            }
        }
        
        return segments
    
    def draw_stress_zone(self, image, segment_data, segment_name):
        """Draw stress visualization for a body segment"""
        stress_level = min(segment_data['stress'], 1.0)  # Cap at 1.0
        center = segment_data['center']
        
        # Color mapping: Green (good) -> Yellow (caution) -> Red (bad)
        if stress_level < 0.3:
            color = (0, int(255 * (1 - stress_level)), 0)  # Green zone
        elif stress_level < 0.7:
            color = (0, 255, int(255 * (stress_level - 0.3) / 0.4))  # Yellow zone
        else:
            color = (0, int(255 * (1 - stress_level)), 255)  # Red zone
        
        # Draw gradient circle for stress visualization
        radius = int(30 + stress_level * 20)  # Radius based on stress
        
        # Create gradient effect
        for i in range(radius, 0, -2):
            alpha = (radius - i) / radius
            temp_color = tuple(int(c * alpha) for c in color)
            cv2.circle(image, center, i, temp_color, -1)
        
        # Add stress level text
        if stress_level > 0.2:
            stress_text = f"{stress_level:.1f}"
            font_scale = 0.4
            thickness = 1
            text_size = cv2.getTextSize(stress_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            
            # Text background
            cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 2), 
                         (text_x + text_size[0] + 2, text_y + 2), (0, 0, 0), -1)
            cv2.putText(image, stress_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    def draw_enhanced_landmarks(self, image, landmarks, stress_data, width, height):
        """Draw enhanced pose landmarks with stress color coding"""
        points = landmarks.pose_landmarks.landmark
        
        # Define key joints and their stress levels
        joint_stress = {
            self.mp_pose.PoseLandmark.LEFT_KNEE.value: stress_data.get('left_knee_stress', 0),
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value: stress_data.get('right_knee_stress', 0),
            self.mp_pose.PoseLandmark.LEFT_HIP.value: stress_data.get('left_hip_stress', 0),
            self.mp_pose.PoseLandmark.RIGHT_HIP.value: stress_data.get('right_hip_stress', 0),
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value: stress_data.get('left_ankle_stress', 0),
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value: stress_data.get('right_ankle_stress', 0),
        }
        
        # Draw enhanced landmarks
        for landmark_idx, stress in joint_stress.items():
            if landmark_idx < len(points):
                x = int(points[landmark_idx].x * width)
                y = int(points[landmark_idx].y * height)
                
                # Color based on stress level
                if stress < 0.3:
                    color = (0, 255, 0)  # Green
                elif stress < 0.7:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw joint marker
                cv2.circle(image, (x, y), 8, color, -1)
                cv2.circle(image, (x, y), 10, (255, 255, 255), 2)
        
        # Draw skeleton connections with stress-based coloring
        connections = [
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value),
            (self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_HIP.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value),
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                start_x = int(points[start_idx].x * width)
                start_y = int(points[start_idx].y * height)
                end_x = int(points[end_idx].x * width)
                end_y = int(points[end_idx].y * height)
                
                # Average stress for connection
                start_stress = joint_stress.get(start_idx, 0)
                end_stress = joint_stress.get(end_idx, 0)
                avg_stress = (start_stress + end_stress) / 2
                
                # Connection color based on average stress
                if avg_stress < 0.3:
                    conn_color = (0, 255, 0)
                elif avg_stress < 0.7:
                    conn_color = (0, 255, 255)
                else:
                    conn_color = (0, 0, 255)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), conn_color, 3)
    
    def draw_form_indicators(self, image, prediction, width, height):
        """Draw overall form quality indicators"""
        # Form status indicator
        if prediction is not None:
            status_text = "GOOD FORM" if prediction == 1 else "POOR FORM"
            status_color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
            
            # Background rectangle for status
            cv2.rectangle(image, (10, 10), (200, 50), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (200, 50), status_color, 2)
            cv2.putText(image, status_text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Add legend
        legend_y = height - 120
        cv2.rectangle(image, (10, legend_y), (150, height - 10), (0, 0, 0), -1)
        cv2.putText(image, "Stress Level:", (15, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Color legend
        cv2.circle(image, (25, legend_y + 35), 8, (0, 255, 0), -1)
        cv2.putText(image, "Low", (40, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        cv2.circle(image, (25, legend_y + 55), 8, (0, 255, 255), -1)
        cv2.putText(image, "Medium", (40, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        cv2.circle(image, (25, legend_y + 75), 8, (0, 0, 255), -1)
        cv2.putText(image, "High", (40, legend_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

class SquatFormClassifier:
    """Class to handle squat form classification"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_training_data(self, n_samples=1000):
        """Generate synthetic training data for squat form"""
        np.random.seed(42)
        
        # Good form characteristics
        good_samples = []
        for _ in range(n_samples // 2):
            # Good squat: knees 90-120°, hips 90-110°, minimal spine deviation
            left_knee = np.random.uniform(90, 120)
            right_knee = np.random.uniform(90, 120)
            left_hip = np.random.uniform(90, 110)
            right_hip = np.random.uniform(90, 110)
            spine = np.random.uniform(0.02, 0.08)  # Minimal deviation
            left_alignment = np.random.uniform(0.02, 0.06)
            right_alignment = np.random.uniform(0.02, 0.06)
            depth = np.random.uniform(0.15, 0.25)  # Good depth
            
            good_samples.append([left_knee, right_knee, left_hip, right_hip, 
                               spine, left_alignment, right_alignment, depth])
        
        # Bad form characteristics
        bad_samples = []
        for _ in range(n_samples // 2):
            # Bad squat: extreme angles, poor alignment
            left_knee = np.random.choice([
                np.random.uniform(60, 89),   # Too shallow
                np.random.uniform(121, 160)  # Too deep/unstable
            ])
            right_knee = np.random.choice([
                np.random.uniform(60, 89),
                np.random.uniform(121, 160)
            ])
            left_hip = np.random.choice([
                np.random.uniform(60, 89),   # Not enough hip hinge
                np.random.uniform(111, 140)  # Too much forward lean
            ])
            right_hip = np.random.choice([
                np.random.uniform(60, 89),
                np.random.uniform(111, 140)
            ])
            spine = np.random.uniform(0.1, 0.3)    # Poor spine alignment
            left_alignment = np.random.uniform(0.08, 0.2)   # Knees cave in/out
            right_alignment = np.random.uniform(0.08, 0.2)
            depth = np.random.choice([
                np.random.uniform(0.05, 0.14),  # Too shallow
                np.random.uniform(0.26, 0.4)    # Too deep
            ])
            
            bad_samples.append([left_knee, right_knee, left_hip, right_hip,
                              spine, left_alignment, right_alignment, depth])
        
        # Create dataset
        X = np.array(good_samples + bad_samples)
        y = np.array([1] * len(good_samples) + [0] * len(bad_samples))  # 1=good, 0=bad
        
        return X, y
    
    def train_model(self):
        """Train the squat form classification model"""
        print("Generating training data...")
        X, y = self.generate_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Bad Form', 'Good Form']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Bad Form', 'Good Form'],
                   yticklabels=['Bad Form', 'Good Form'])
        plt.title('Squat Form Classification - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        self.is_trained = True
        return accuracy, cm, plt
    
    def predict_form(self, features):
        """Predict form quality from features"""
        if not self.is_trained or features is None:
            return None, 0.0
        
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        return prediction, confidence
    
    def get_feedback(self, features):
        """Get detailed feedback based on features"""
        if features is None:
            return "Cannot detect pose properly"
        
        feedback = []
        
        # Check knee angles
        left_knee, right_knee = features[0], features[1]
        if left_knee < 90 or right_knee < 90:
            feedback.append("⚠️ Squat deeper - knees should reach 90°")
        elif left_knee > 120 or right_knee > 120:
            feedback.append("⚠️ You're going too low - control the descent")
        
        # Check spine alignment
        spine_deviation = features[4]
        if spine_deviation > 0.1:
            feedback.append("⚠️ Keep your spine neutral - avoid leaning forward")
        
        # Check knee alignment
        left_alignment, right_alignment = features[5], features[6]
        if left_alignment > 0.08 or right_alignment > 0.08:
            feedback.append("⚠️ Keep knees aligned over toes")
        
        # Check squat depth
        depth = features[7]
        if depth < 0.15:
            feedback.append("⚠️ Squat deeper for better muscle activation")
        
        return " | ".join(feedback) if feedback else "✅ Good form!"
    
    def calculate_stress_levels(self, features):
        """Calculate stress levels for different body parts based on form analysis"""
        if features is None:
            return {}
        
        stress_data = {}
        
        # Knee stress based on angle deviation from optimal range (90-120°)
        left_knee_angle, right_knee_angle = features[0], features[1]
        optimal_knee_min, optimal_knee_max = 90, 120
        
        left_knee_stress = 0
        if left_knee_angle < optimal_knee_min:
            left_knee_stress = (optimal_knee_min - left_knee_angle) / optimal_knee_min
        elif left_knee_angle > optimal_knee_max:
            left_knee_stress = (left_knee_angle - optimal_knee_max) / optimal_knee_max
        
        right_knee_stress = 0
        if right_knee_angle < optimal_knee_min:
            right_knee_stress = (optimal_knee_min - right_knee_angle) / optimal_knee_min
        elif right_knee_angle > optimal_knee_max:
            right_knee_stress = (right_knee_angle - optimal_knee_max) / optimal_knee_max
        
        # Hip stress based on angle deviation from optimal range (90-110°)
        left_hip_angle, right_hip_angle = features[2], features[3]
        optimal_hip_min, optimal_hip_max = 90, 110
        
        left_hip_stress = 0
        if left_hip_angle < optimal_hip_min:
            left_hip_stress = (optimal_hip_min - left_hip_angle) / optimal_hip_min
        elif left_hip_angle > optimal_hip_max:
            left_hip_stress = (left_hip_angle - optimal_hip_max) / optimal_hip_max
        
        right_hip_stress = 0
        if right_hip_angle < optimal_hip_min:
            right_hip_stress = (optimal_hip_min - right_hip_angle) / optimal_hip_min
        elif right_hip_angle > optimal_hip_max:
            right_hip_stress = (right_hip_angle - optimal_hip_max) / optimal_hip_max
        
        # Spine stress based on alignment deviation
        spine_deviation = features[4]
        spine_stress = min(spine_deviation * 10, 1.0)  # Normalize to 0-1
        
        # Ankle/alignment stress
        left_alignment, right_alignment = features[5], features[6]
        left_ankle_stress = min(left_alignment * 10, 1.0)
        right_ankle_stress = min(right_alignment * 10, 1.0)
        
        stress_data = {
            'left_knee_stress': min(left_knee_stress, 1.0),
            'right_knee_stress': min(right_knee_stress, 1.0),
            'left_hip_stress': min(left_hip_stress, 1.0),
            'right_hip_stress': min(right_hip_stress, 1.0),
            'spine_stress': spine_stress,
            'left_ankle_stress': left_ankle_stress,
            'right_ankle_stress': right_ankle_stress
        }
        
        return stress_data

class FitnessFormChecker:
    """Main application class"""
    
    def __init__(self):
        self.pose_extractor = PoseExtractor()
        self.classifier = SquatFormClassifier()
        self.cap = None
    
    def train_system(self):
        """Train the form classification system"""
        return self.classifier.train_model()
    
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("Cannot access camera")
            return False
        return True
    
    def stop_camera(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
    
    def process_frame(self, frame):
        """Process a single frame for form analysis with heatmap overlay"""
        # Extract pose
        results = self.pose_extractor.extract_landmarks(frame)
        
        # Extract features
        features = self.pose_extractor.extract_squat_features(results)
        
        # Get prediction
        prediction, confidence = self.classifier.predict_form(features)
        
        # Get feedback
        feedback = self.classifier.get_feedback(features)
        
        # Calculate stress levels for heatmap
        stress_data = self.classifier.calculate_stress_levels(features)
        
        # Draw heatmap overlay on the frame
        frame_with_heatmap = self.pose_extractor.draw_heatmap_overlay(
            frame, results, stress_data, prediction)
        
        return frame_with_heatmap, prediction, confidence, feedback, features, stress_data

def create_streamlit_app():
    """Create Streamlit interface"""
    st.set_page_config(page_title="AI Fitness Form Checker", layout="wide")
    
    st.title("🏋️‍♂️ AI-Powered Fitness Form Checker")
    st.markdown("Real-time squat form analysis using MediaPipe and Machine Learning")
    
    # Initialize session state
    if 'form_checker' not in st.session_state:
        st.session_state.form_checker = FitnessFormChecker()
        st.session_state.trained = False
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Training section
    st.sidebar.subheader("1. Model Training")
    if st.sidebar.button("Train Model", type="primary"):
        with st.spinner("Training model... This may take a moment."):
            accuracy, cm, plt_fig = st.session_state.form_checker.train_system()
            st.session_state.trained = True
            
            st.success(f"✅ Model trained! Accuracy: {accuracy:.1%}")
            
            # Show confusion matrix
            st.subheader("Model Performance")
            st.pyplot(plt_fig)
    
    # Camera section
    st.sidebar.subheader("2. Camera Control")
    camera_on = st.sidebar.checkbox("Enable Camera")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Video Feed")
        video_placeholder = st.empty()
        
        if camera_on and st.session_state.trained:
            if st.session_state.form_checker.start_camera():
                
                # Real-time processing
                while camera_on:
                    ret, frame = st.session_state.form_checker.cap.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        break
                    
                    # Process frame
                    processed_frame, prediction, confidence, feedback, features, stress_data = \
                        st.session_state.form_checker.process_frame(frame)
                    
                    # Display frame with heatmap
                    video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                    
                    # Update sidebar with feedback
                    with col2:
                        st.subheader("Real-time Feedback")
                        if prediction is not None:
                            if prediction == 1:
                                st.success("✅ Good Form!")
                            else:
                                st.error("❌ Poor Form")
                            
                            st.metric("Confidence", f"{confidence:.1%}")
                            st.info(feedback)
                            
                            # Show feature values
                            if features:
                                st.subheader("Pose Metrics")
                                metrics = [
                                    "Left Knee Angle", "Right Knee Angle",
                                    "Left Hip Angle", "Right Hip Angle",
                                    "Spine Alignment", "Left Knee Align", 
                                    "Right Knee Align", "Squat Depth"
                                ]
                                
                                for i, (metric, value) in enumerate(zip(metrics, features)):
                                    st.metric(metric, f"{value:.1f}")
                            
                            # Show stress levels with enhanced visualization
                            if stress_data:
                                st.subheader("Real-Time Stress Analysis")
                                
                                # Create and display real-time heatmap
                                heatmap_fig = create_real_time_heatmap_display(stress_data)
                                if heatmap_fig:
                                    st.pyplot(heatmap_fig)
                                
                                # Detailed stress breakdown
                                st.subheader("Detailed Stress Breakdown")
                                col_left, col_right = st.columns(2)
                                
                                with col_left:
                                    for i, (body_part, stress_level) in enumerate(list(stress_data.items())[:4]):
                                        part_name = body_part.replace('_', ' ').title()
                                        if stress_level < 0.3:
                                            st.success(f"🟢 {part_name}: {stress_level:.2f}")
                                        elif stress_level < 0.7:
                                            st.warning(f"🟡 {part_name}: {stress_level:.2f}")
                                        else:
                                            st.error(f"🔴 {part_name}: {stress_level:.2f}")
                                
                                with col_right:
                                    for i, (body_part, stress_level) in enumerate(list(stress_data.items())[4:]):
                                        part_name = body_part.replace('_', ' ').title()
                                        if stress_level < 0.3:
                                            st.success(f"🟢 {part_name}: {stress_level:.2f}")
                                        elif stress_level < 0.7:
                                            st.warning(f"🟡 {part_name}: {stress_level:.2f}")
                                        else:
                                            st.error(f"🔴 {part_name}: {stress_level:.2f}")
                        else:
                            st.warning("No pose detected")
                    
                    # Small delay
                    import time
                    time.sleep(0.1)
                
                st.session_state.form_checker.stop_camera()
        
        elif camera_on and not st.session_state.trained:
            st.warning("⚠️ Please train the model first!")
        
        elif not camera_on:
            st.info("📹 Enable camera to start form checking")
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        **How to use:**
        1. Click 'Train Model' to prepare the AI
        2. Enable camera for live analysis
        3. Perform squats in front of the camera
        4. Get real-time form feedback
        
        **Squat Form Tips:**
        - Keep feet shoulder-width apart
        - Lower until thighs are parallel to ground
        - Keep knees aligned over toes
        - Maintain neutral spine
        - Control the movement
        
        **Heatmap Legend:**
        - 🟢 Green: Good form, low stress
        - 🟡 Yellow: Caution, medium stress  
        - 🔴 Red: Poor form, high stress
        - Circles show stress intensity
        - Enhanced joints show problem areas
        """)
        
        if not st.session_state.trained:
            st.info("💡 Train the model to get started!")

# Additional utility functions
def save_model(form_checker, filename="squat_model.pkl"):
    """Save trained model to file"""
    with open(filename, 'wb') as f:
        pickle.dump({
            'model': form_checker.classifier.model,
            'scaler': form_checker.classifier.scaler
        }, f)

def load_model(form_checker, filename="squat_model.pkl"):
    """Load trained model from file"""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            form_checker.classifier.model = data['model']
            form_checker.classifier.scaler = data['scaler']
            form_checker.classifier.is_trained = True
        return True
    except FileNotFoundError:
        return False

# Heat map visualization for muscle stress (optional enhancement)
def create_stress_heatmap(features, prediction):
    """Create a simple heatmap showing areas of concern"""
    if features is None:
        return None
    
    # Create a simple body outline with stress indicators
    stress_map = {
        'knees': max(abs(features[0] - 105), abs(features[1] - 105)) / 105,  # Ideal ~105°
        'hips': max(abs(features[2] - 100), abs(features[3] - 100)) / 100,   # Ideal ~100°
        'spine': features[4] * 10,  # Spine deviation
        'alignment': (features[5] + features[6]) * 5  # Knee alignment
    }
    
    return stress_map

if __name__ == "__main__":
    # Run the Streamlit app
    create_streamlit_app()