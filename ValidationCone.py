import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="FreeSans", size=18)
from sklearn.cluster import DBSCAN
from sklearn import linear_model

class ValidationCone:
    def __init__(self, SRC):
        self.SRC = SRC
        cap = cv2.VideoCapture(self.SRC+"data.mp4") # Read in source video
        self.dim = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))] # Size of video frame
        self.duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
        self.fps = 30.0

    def find_wheels(self):
        sys.stdout.write("Finding vehicles...\n")
        cap = cv2.VideoCapture(self.SRC+"data.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('demo.avi',fourcc, self.fps,  (self.dim[1],self.dim[0]))
        wheel_centers = []
        ret, frame = cap.read()
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        idx = 0
        while(1):
            ret, frame = cap.read()
            if ret == False:
                break
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sys.stdout.write("Processing frame %d...\r" % idx)
            sys.stdout.flush()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray_float = gray.astype(np.float)
            bg_idx = idx / 900 + 1
            bg = cv2.imread("background/bg_%02d.png" % bg_idx, 0).astype(np.float)
            fg =  np.abs(gray_float - bg).astype(np.uint8)
            if np.sum(fg>32) > 50000:
                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 
                                                    0.5, 8, 32, 3, 7, 1.5, 
                                                    cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                #pos_mask = np.logical_or(ang < 0.2*np.pi, ang > 1.8*np.pi)
                kernel = np.ones((2,2),np.uint8)
                mag = cv2.erode(mag,kernel,iterations = 1)
                pos_mask = mag > 1
                neg_mask = np.logical_not(pos_mask)
                gray_ = np.array(gray)
                gray_[neg_mask] = 0
                circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,1,
                                           param1=192,param2=28,
                                           minRadius=self.dim[0]/5,
                                           maxRadius=self.dim[0]/3)
                try:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0,:]:
                        if pos_mask[i[1], i[0]] == True:
                            # store the center
                            wheel_centers.append([idx, i[0], i[1], i[2]])
                            # draw the outer circle
                            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                            # draw the center of the circle
                            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                except:
                    pass
                cv2.imshow("Mask", gray_)
                cv2.waitKey(100)

            cv2.imshow("Detected Circles", frame)
            out.write(frame)
            cv2.waitKey(1)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            idx += 1
            prvs = next
        wheel_centers = np.asarray(wheel_centers)
        np.save(self.SRC+"wheel_centers.npy", wheel_centers)
        sys.stdout.write("\nDone.\n")
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    def find_dynamics(self):
        sys.stdout.write("Detecting vehicles...\n")
        wheel_centers = np.load(self.SRC+"wheel_centers.npy")
        wheel_centers = wheel_centers[0:,...].astype(np.float)
        wheel_centers[:,0] *= 4.
        wheel_centers[:,1] /= 24.

        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(wheel_centers[:,0]/4., wheel_centers[:,1]*24., 'g.')
        ax2 = fig.add_subplot(2,1,2)

        dbscan = DBSCAN(eps=16, min_samples=4)  
        dbscan.fit(wheel_centers)
        labels = dbscan.labels_
        labels_unique = np.unique(labels)
        if -1 in labels_unique:
            num = len(labels_unique)-1
        else:
            num = len(labels_unique)
        print "Found %d vehicles." % num
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels_unique)))
        for label, color in zip(labels_unique, colors):
            if label != -1:
                mask = labels == label
                cluster = wheel_centers[mask]
                model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
                X = cluster[:,0].reshape(-1, 1) / 4.
                y = cluster[:,1].reshape(-1, 1) * 24.
                model_ransac.fit(y, X)
                plt.plot(X, y, '.', color=color)
                y_ransac = np.arange(0, 640).reshape(-1, 1)
                X_ransac = model_ransac.predict(y_ransac)
                velocity = ((np.max(y_ransac) - np.min(y_ransac)) / 
                            (np.max(X_ransac) - np.min(X_ransac)))
                ax2.plot(X_ransac, y_ransac)
                #ax2.set_xlim([np.min(X_ransac), np.min(X_ransac)+15])
                #ax2.set_ylim([0, 640])
                #ax2.set_title("Velocity: %f" % velocity)

        plt.show()

if __name__ == "__main__":
    vc = ValidationCone("./")
    vc.find_wheels()
    vc.find_dynamics()
