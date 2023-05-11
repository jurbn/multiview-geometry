import cv2 as cv
import sys
import numpy as np
import time
import copy

FILE = '../../media/secuencia_c_cam2.avi'


def drawlines(img1, img2, lines, pts1, pts2):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines"""
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt1 = [int(pt) for pt in pt1[0]]
        pt2 = [int(pt) for pt in pt2[0]]
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        print(tuple(pt1))
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def nn_dr(matches, ratio=0.75):
    """Nearest neighbour distance ratio"""
    selected_matches = []
    for matchs in matches:
        if matchs[0].distance / matchs[1].distance < ratio:
            selected_matches.append(matchs[0])
    return selected_matches


def main(args):
    n = 0
    orb = cv.ORB_create()
    bfmatcher = cv.BFMatcher_create(cv.NORM_HAMMING, False)
    cap = cv.VideoCapture(FILE)  # video capture object
    cap.set(cv.CAP_PROP_POS_FRAMES, n)
    ret, frame_og = cap.read()
    if not ret:
        raise Exception("No se encontró el primer frame")
    kp1, des1 = orb.detectAndCompute(frame_og, None)
    while cap.isOpened():
        n += 1
        cap.set(cv.CAP_PROP_POS_FRAMES, n)
        ret, frame_nxt = cap.read()
        if ret:
            # sacamos descriptores y matches!
            frame_og_cp = copy.deepcopy(frame_og)
            kp2, des2 = orb.detectAndCompute(frame_nxt, None)
            matches = bfmatcher.knnMatch(des1,  des2, k=2)
            selected_matches = nn_dr(matches)
            if len(selected_matches) > 10:  # if not enough matches, dont draw anything
                src_pts = np.float32([kp1[m.queryIdx].pt for m in selected_matches]).reshape(-1, 1, 2)  # kp of src
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in selected_matches]).reshape(-1, 1, 2)  # kp of sdt
                F, mask = cv.findFundamentalMat(src_pts, dst_pts, method=cv.FM_RANSAC, confidence=0.9999);  # confidence sets the number of iterations
                src_pts = src_pts[mask.ravel() == 1]
                dst_pts = dst_pts[mask.ravel() == 1]
                try:
                    epilines = cv.computeCorrespondEpilines(dst_pts, 2, F)
                    epilines = epilines.reshape(-1, 3)
                    drawed_og, _ = drawlines(frame_og_cp, frame_nxt, epilines, src_pts, dst_pts)
                except Exception:
                    drawed_og = frame_og_cp
            else:
                drawed_og = frame_og_cp
            img_matches = cv.drawMatches(drawed_og, [], frame_nxt, [], [], None, matchColor = (0,255,0))

            cv.namedWindow("Matches seleccionados", cv.WINDOW_NORMAL)
            cv.imshow("Matches seleccionados", img_matches)

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
        #time.sleep(0.1)


if __name__ == '__main__':
    main(sys.argv)