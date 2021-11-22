import os
import os.path as osp
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from ..baseExtractor import baseExtractor

import mediapipe as mp


class mediapipeExtractor(baseExtractor):
    """
    Video feature extractor using MediaPipe. 
    Ref: https://mediapipe.dev/
    """
    def __init__(self, config, logger):
        try:
            logger.info("Initializing MediaPipe video feature extractor...")
            super().__init__(config, logger)
            self.args = self.config['args']
            if self.args['visualize']: # drawing utilities
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                self.drawing_spec = self.mp_drawing.DrawingSpec(
                    thickness=1, circle_radius=1
                )
            if 'holistic' in self.args:
                self.kwargs = self.args['holistic']
                self.method = mp.solutions.holistic.Holistic
            elif 'face_mesh' in self.args:
                self.kwargs = self.args['face_mesh']
                self.kwargs['max_num_faces'] = 1
                self.method = mp.solutions.face_mesh.FaceMesh
        except Exception as e:
            self.logger.error("Failed to initialize mediapipeExtractor.")
            raise e

    def extract(self, img_dir, video_name=None):
        """
        Function:
            Extract features from video file using MediaPipe.

        Parameters:
            img_dir: path to directory of images.
            video_name: video name used to save annotation images.

        Returns:
            video_result: extracted video features in numpy array.
        """
        try:
            video_result = []
            with self.method(static_image_mode=False, **self.kwargs) as method:
                for image_path in sorted(glob(osp.join(img_dir, '*.bmp'))):
                    name = Path(image_path).stem
                    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    result = method.process(image)
                    if 'holistic' in self.args:
                        if not result.face_landmarks:
                            self.logger.debug(f"No face detected in {image_path}.")
                            continue
                        if self.args['visualize']:
                            assert video_name is not None, \
                                "video_name should be passed in order to save annotation images."
                            annotated_image = image.copy()
                            condition = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1
                            bg_image = np.zeros(image.shape, dtype=np.uint8)
                            annotated_image = np.where(condition, annotated_image, bg_image)
                            self.mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=result.face_landmarks,
                                connections=mp.solutions.holistic.FACEMESH_TESSELATION,
                                landmark_drawing_spec = None,
                                connection_drawing_spec = self.mp_drawing_styles
                                .get_default_face_mesh_tesselation_style()
                            )
                            self.mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=result.pose_landmarks,
                                connections=mp.solutions.holistic.POSE_CONNECTIONS,
                                landmark_drawing_spec = self.mp_drawing_styles.
                                get_default_pose_landmarks_style()
                            )
                            self.mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=result.left_hand_landmarks,
                                connections=mp.solutions.holistic.HAND_CONNECTIONS,
                                landmark_drawing_spec = self.mp_drawing_styles.
                                get_default_hand_landmarks_style()
                            )
                            self.mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=result.right_hand_landmarks,
                                connections=mp.solutions.holistic.HAND_CONNECTIONS,
                                landmark_drawing_spec = self.mp_drawing_styles.
                                get_default_hand_landmarks_style()
                            )
                            os.makedirs(osp.join(self.args['visualize_dir'], video_name), exist_ok=True)
                            cv2.imwrite(osp.join(self.args['visualize_dir'], video_name, name + '.jpg'),
                                        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                        
                        res_image = []
                        for landmark in result.face_landmarks.landmark:
                            res_image.append(landmark.x)
                            res_image.append(landmark.y)
                            res_image.append(landmark.z)
                        if result.pose_landmarks:
                            for landmark in result.pose_landmarks.landmark:
                                res_image.append(landmark.x)
                                res_image.append(landmark.y)
                                res_image.append(landmark.z)
                        else:
                            res_image.extend([0]*33*3)
                        if result.left_hand_landmarks:
                            for landmark in result.left_hand_landmarks.landmark:
                                res_image.append(landmark.x)
                                res_image.append(landmark.y)
                                res_image.append(landmark.z)
                        else:
                            res_image.extend([0]*21*3)
                        if result.right_hand_landmarks:
                            for landmark in result.right_hand_landmarks.landmark:
                                res_image.append(landmark.x)
                                res_image.append(landmark.y)
                                res_image.append(landmark.z)
                        else:
                            res_image.extend([0]*21*3)
                        video_result.append(res_image)
                    elif 'face_mesh' in self.args:
                        if not result.multi_face_landmarks:
                            self.logger.debug(f"No face detected in {image_path}.")
                            continue
                        if self.args['visualize']:
                            assert video_name is not None, \
                                "video_name should be passed in order to save annotation images."
                            annotated_image = image.copy()
                            self.mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=result.multi_face_landmarks[0],
                                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles
                                .get_default_face_mesh_tesselation_style()
                            )
                            self.mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=result.multi_face_landmarks[0],
                                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles
                                .get_default_face_mesh_contours_style()
                            )
                            os.makedirs(osp.join(self.args['visualize_dir'], video_name), exist_ok=True)
                            cv2.imwrite(osp.join(self.args['visualize_dir'], video_name, name + '.jpg'),
                                        cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                        res_image = []
                        for landmark in result.multi_face_landmarks[0].landmark:
                            res_image.append(landmark.x)
                            res_image.append(landmark.y)
                            res_image.append(landmark.z)
                        video_result.append(res_image)
                video_result = np.array(video_result)
                return video_result

        except Exception as e:
            self.logger.error(f"Failed to extract video features with MediaPipe from {video_name}.")
            raise e
