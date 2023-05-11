import cv2 as cv
import sys
import numpy as np
import random
import time

FILE = '../../media/secuencia_b_cam2.avi'

def nn_dr(matches, ratio=0.9):
    """Nearest neighbour distance ratio"""
    selected_matches = []
    for matchs in matches:
        if matchs[0].distance / matchs[1].distance < ratio:
            selected_matches.append(matchs[0])
    return selected_matches


def main(args):
    n = 0
    sift = cv.SIFT_create()
    bfmatcher = cv.BFMatcher_create(cv.NORM_L2, False)
    cap = cv.VideoCapture(FILE)  # video capture object
    cap.set(cv.CAP_PROP_POS_FRAMES, n)
    ret, frame_og = cap.read()
    if not ret:
        raise Exception("No se encontró el primer frame")
    kp1, des1 = sift.detectAndCompute(frame_og, None)
    while cap.isOpened():
        n += 1
        cap.set(cv.CAP_PROP_POS_FRAMES, n)
        ret, frame_nxt = cap.read()
        if ret:
            # sacamos descriptores y matches!
            kp2, des2 = sift.detectAndCompute(frame_nxt, None)
            matches = bfmatcher.knnMatch(des1,  des2, k=2)
            selected_matches = nn_dr(matches)
            print('number of matches: {}'.format(len(selected_matches)))
            src_pts = np.float32([kp1[m.queryIdx].pt for m in selected_matches]).reshape(-1, 2)  # kp of src
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in selected_matches]).reshape(-1, 2)  # kp of sdt
            M, mask = cv.findHomography(src_pts, dst_pts, method=cv.RANSAC)
            Minv = np.linalg.inv(M)
            dst1 = cv.warpPerspective(frame_og, np.eye(3), (1024, 1024))
            dst2 = cv.warpPerspective(frame_nxt, Minv, (1024, 1024))
            #dst1 = cv.warpPerspective(frame_og, M, (1024, 1024))
            #dst2 = cv.warpPerspective(frame_nxt, np.eye(3), (1024, 1024))
            weighted = cv.addWeighted(dst1, 0.5, dst2, 0.5, 0.0)
            cv.imshow("Matches seleccionados", weighted)

            if cv.waitKey(1) == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                print('The video was stopped')
                exit()
            print("Pulsa q para salir")
        else:
            cap.release()
            cv.destroyAllWindows()
            print('The video ended!')
            exit()


if __name__ == '__main__':
    main(sys.argv)