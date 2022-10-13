import cv2,pdb
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

def plot3D(landmarks):
    connect_lm = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,23],[12,24],[23,25],[24,26],[26,28],[28,30],[28,32],[25,27],\
        [27,31],[27,29],[29,31],[30,32],[30,32],[11,13],[13,15],[15,21],[15,19],[15,17],[17,19],[12,14],[14,16],[16,22],[16,18],[16,20],[18,20]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # for point in landmarks:
    #     x = point[0]
    #     y = point[1]
    #     z = point[2]
    #     ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlim(xmin = -1, xmax = 1)
    ax.set_ylim(ymin = -1, ymax = 1)
    ax.set_zlim(zmin = -1, zmax = 1)
    for lamb in connect_lm:
        l = landmarks[lamb[0]]
        r = landmarks[lamb[1]]
        plt.plot([l[0],r[0]],[l[1],r[1]],[l[2],r[2]],color='black',marker='o',markerfacecolor='red', markersize=5)
    
    img_w, img_h = fig.canvas.get_width_height()
    fig.canvas.draw()
    img_vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    plt.close()

    return img_vis


def KFsmoother(measurements):
    '''
    measurements : N*3  observations
    '''
    transition_m = [[1,0,0,0.03,0,0],[0,1,0,0,0.03,0],[0,0,1,0,0,0.03],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
    observation_m = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]]


    kf = KalmanFilter(transition_matrices = transition_m, observation_matrices = observation_m)
    kf = kf.em(measurements, n_iter=5)

    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

    return smoothed_state_means

def main():
    ## read data
    video_path = './data/dance_test.mp4'
    out_path1 = './result/origin.mp4'
    out_path2 = './result/KF.mp4'
    cap = cv2.VideoCapture(video_path)
    datas_pose = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path1, fourcc, 30,(640, 480))
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if index == 10:
            break
        if ret:
            index += 1
            print(index)
            # cv2.imshow('frame', frame)
            # key = cv2.waitKey(25)
            # if key == ord('q'): #当键入空格或者q时，则退出while循环
            #     break
            landmarks = []
            with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
                # Convert the BGR image to RGB and process it with MediaPipe Pose.
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for id, lm in enumerate(results.pose_world_landmarks.landmark):
                   landmarks.append([lm.x,lm.z,-lm.y])

                datas_pose.append(landmarks)

                ## Plot pose world landmarks.
                # mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
                img_result = plot3D(landmarks)
                writer.write(img_result)
                # pdb.set_trace()
                # Draw pose landmarks.
                # annotated_image = frame.copy()
                # mp_drawing.draw_landmarks(
                #     annotated_image,
                #     results.pose_landmarks,
                #     mp_pose.POSE_CONNECTIONS,
                #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # cv2.imshow('frame',annotated_image)
                # key = cv2.waitKey(0)
                # Plot pose world landmarks.
                # mp_drawing.plot_landmarks(
                #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    cap.release()
    writer.release()
    ## KF smoother
    datas_pose_np = np.array(datas_pose)
    new_data = datas_pose_np.reshape(33,-1,3)
    new_list_pose = []
    for i in new_data:
        new_pose = KFsmoother(i)  ## Kalman smoother
        new_list_pose.append(new_pose[:,:3])
    
    new_list_pose2 = np.array(new_list_pose).reshape(-1,33,3)
    writer = cv2.VideoWriter(out_path2, fourcc, 30,(640, 480))
    for i in range(new_list_pose2.shape[0]):
        img_result = plot3D(list(new_list_pose2[i]))
        writer.write(img_result)
    writer.release()


if __name__ == '__main__':
    main()