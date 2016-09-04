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
    # Initialize the validation class.
    def __init__(self, SRC):
        self.SRC = SRC
        cap = cv2.VideoCapture(self.SRC+"data.mp4") # Read in source video
        self.dim = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))] # Size of video frame
        self.duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
        self.fps = 30.0

    # Fine the location and size of wheels in every frame 
    # using Hough circle transformation.
    def find_wheels(self):
        sys.stdout.write("Finding vehicles...\n")
        cap = cv2.VideoCapture(self.SRC+"data.mp4")
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('demo.avi',fourcc, self.fps,  (self.dim[1],self.dim[0]))
        wheel_centers = []
        idx = 0.0
        while(1):
            ret, frame = cap.read()
            if ret == False:
                break
            sys.stdout.write("Processing frame %d...\r" % idx)
            sys.stdout.flush()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            ret, black = cv2.threshold(gray,64,255,cv2.THRESH_BINARY)
            black = np.uint8(black)
            kernel = np.ones((4,4),np.uint8)
            black = cv2.dilate(black,kernel,iterations = 2)
            if np.sum(black < 127) > 7200:
                circles = cv2.HoughCircles(black,cv2.HOUGH_GRADIENT,1,320,
                                           param1=64,param2=8,
                                           minRadius=self.dim[0]/5,
                                           maxRadius=self.dim[0]/3)
                try:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0,:]:
                        # store the center
                        wheel_centers.append([idx/30., i[0], i[1], i[2]])
                        # draw the outer circle
                        cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                        # draw the center of the circle
                        cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                except:
                    pass
                #cv2.imshow("Black", black)
                #cv2.waitKey(100)
            #cv2.imshow("Detected Circles", frame)
            #out.write(frame)
            #k = cv2.waitKey(1) & 0xff
            #if k == 27:
            #    break
            idx += 1
            prvs = next
        wheel_centers = np.asarray(wheel_centers)
        np.save(self.SRC+"wheel_centers.npy", wheel_centers)
        sys.stdout.write("\nDone.\n")
        #out.release()
        cap.release()
        cv2.destroyAllWindows()

    # Process the raw wheel location and size data. Cluster
    # data points that belong to the same wheel into one group.
    # LINEARLY interpolate and extrapolate missing data when necessary.
    def find_dynamics(self):
        sys.stdout.write("Detecting vehicles...\n")
        wheel_centers = np.load(self.SRC+"wheel_centers.npy")
        wheel_centers = wheel_centers.astype(np.float)
        wheel_centers[:,0] *= 4.*30.
        wheel_centers[:,1] /= 24.
        wheel_centers[:,2] /= 2.
        wheel_centers[:,3] /= 2.
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(wheel_centers[:,0]/(4.*30.), wheel_centers[:,1]*24., 'g.')
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(wheel_centers[:,0]/(4.*30.), wheel_centers[:,1]*24., 'g.')
        dbscan = DBSCAN(eps=15, min_samples=4)  
        dbscan.fit(wheel_centers)
        labels = dbscan.labels_
        labels_unique = np.unique(labels)
        if -1 in labels_unique:
            num = len(labels_unique)-1
        else:
            num = len(labels_unique)
        print "Found %d wheels." % num
        wheel_dynamics = []
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels_unique)))
        for label, color in zip(labels_unique, colors):
            mask = labels == label
            cluster = wheel_centers[mask]
            t = cluster[:,0].reshape(-1, 1) / (4.*30.)
            x = cluster[:,1].reshape(-1, 1) * 24.
            y = cluster[:,2].reshape(-1, 1) * 2.
            r = np.median(cluster[:,3]) * 2.
            if label != -1:
                model_x = linear_model.RANSACRegressor(linear_model.LinearRegression())
                model_x.fit(x, t)
                model_y = linear_model.RANSACRegressor(linear_model.LinearRegression())
                model_y.fit(t, y)
                ax2.plot(t, x, '.', color=color)
                x_ransac = np.arange(640).reshape(-1, 1)
                t_ransac = model_x.predict(x_ransac).reshape(-1, 1)
                y_ransac = model_y.predict(t_ransac).reshape(-1, 1)
                r_median = np.asarray([r]*len(x_ransac)).reshape(-1, 1)
                velocity = ((np.max(x_ransac) - np.min(x_ransac)) / 
                            (np.max(t_ransac) - np.min(t_ransac)))
                ax2.plot(t_ransac, x_ransac)
                wheel_dynamics.append([np.hstack((t_ransac, x_ransac, 
                                                  y_ransac, r_median))])
            else:
                ax2.plot(t, x, '*', color='k')
        wheel_dynamics = np.asarray(wheel_dynamics)
        wheel_dynamics = np.squeeze(wheel_dynamics)
        np.save("wheel_dynamics.npy", wheel_dynamics)
        plt.show()

    # Visualize the processed dynamics by visually plotting it
    # on the raw video. (Somewhat buggy still. Needs to be fixed soon.)
    def inspect_dynamics(self):
        print "Inspecting detected wheel dynamics..."
        wheel_dynamics = np.load(self.SRC+"wheel_dynamics.npy")
        cap = cv2.VideoCapture(self.SRC+"data.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('demo.avi',fourcc, self.fps,  (self.dim[1],self.dim[0]))
        time = 0.0
        while(1):
            for ID, dynamics in enumerate(wheel_dynamics):
                duration = [min(dynamics[:,0]), max(dynamics[:,0])]
                while time < duration[0]:
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    cv2.imshow("Frame", frame)
                    cv2.waitKey(1)
                    out.write(frame)
                    time += 1.0/30.
                while time < duration[1]:
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    circle = dynamics[np.argmin(np.abs(dynamics[:,0] - time)), 1:].astype(int)
                    # draw the outer circle
                    cv2.circle(frame,(circle[0],circle[1]),circle[2],(0,255,0),2)
                    # draw the center of the circle
                    cv2.circle(frame,(circle[0],circle[1]),2,(0,0,255),3)
                    cv2.imshow("Frame", frame)
                    cv2.waitKey(1)
                    out.write(frame)
                    time += 1.0/30.
                if ret == False:
                    break
            if ret == False:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vc = ValidationCone("./")
    #vc.find_wheels()
    #vc.find_dynamics()
    vc.inspect_dynamics()
