import cv2 as cv
import sys
import numpy as np
import time

FILE = '../../media/secuencia_b_cam2.avi'

def main():
    cap = cv.VideoCapture(FILE)  # video capture object
    n = 0
    cap.set(cv.CAP_PROP_POS_FRAMES, n)
    ret, frame_og = cap.read()
    result = cv.VideoWriter(filename='../../media/homografia.avi', fourcc=cv.VideoWriter_fourcc(*'MJPG'), fps=cap.get(5),
                            frameSize=(int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        n += 1
        cap.set(cv.CAP_PROP_POS_FRAMES, n)
        ret, frame_nxt = cap.read()
        if ret:
            sift = cv.SIFT_create()
            bfmatcher = cv.BFMatcher_create(cv.NORM_L2, False)

            # sacamos descriptores y matches!
            kp1, des1 = sift.detectAndCompute(frame_og, None)
            kp2, des2 = sift.detectAndCompute(frame_nxt, None)
            matches = bfmatcher.knnMatch(des1,  des2, k=2)
            selected_matches = []
            for matchs in matches:
                if matchs[0].distance/matchs[1].distance < 0.75:
                    selected_matches.append(matchs[0])
            if len(selected_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in selected_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in selected_matches]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
                matchesMask = mask.ravel().tolist()
                h, w, _ = frame_og.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                frame_nxt = cv.polylines(frame_nxt, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
                draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
                img_matches = cv.drawMatches(frame_og, kp1, frame_nxt, kp2, selected_matches, None, **draw_params)
            else:
                print('Muy pocos matches :p')
                img_matches = cv.drawMatches(frame_og, [], frame_nxt, [], [], None, matchColor = (0,255,0))

            result.write(img_matches)
            cv.namedWindow("Matches seleccionados", cv.WINDOW_NORMAL)
            cv.imshow("Matches seleccionados", img_matches)


            if cv.waitKey(1) == ord('q'):
                cap.release()
                result.release()
                cv.destroyAllWindows()
                print('The video was stopped')
                #exit()
            print("Pulsa q para salir")
        else:
            cap.release()
            result.release()
            cv.destroyAllWindows()
            print('The video ended!')
            #exit()
    cap.release()
    result.release()
    cv.destroyAllWindows()
    print('The video ended!')
    #exit()


if __name__ == '__main__':
    main()