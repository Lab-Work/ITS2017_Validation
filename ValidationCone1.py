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
from sklearn.ensemble import IsolationForest
from scipy.spatial import ConvexHull

class ValidationCone:
    # Initialize the validation class.
    def __init__(self, SRC, VID, fps, im_coor, re_coor, theta, d):
        self.SRC = SRC # Parent source path
        self.VID = VID # Subpath to video file
        cap = cv2.VideoCapture(self.SRC+self.VID) # Read in source video
        self.dim = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))] # Size of video frame
        self.duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
        self.fps = fps # Frames per second
        self.im_coor = im_coor # Input image coordinates
        self.re_coor = re_coor # Input physical coordinates
        self.theta = theta # Field of view
        self.d = d # Distance from PiCam to outer lane

    # Find the trace of passing vehicles. A revised pipeline using
    # DBSCAN and perhaps some RANSAC-like robust fitting is applied here.
    def find_trace(self):
        trace = [] # Horizontal heat map
        elev = [] # Vertical heat map
        cap = cv2.VideoCapture(self.SRC+self.VID)
        max_duration = self.duration
        time = 0
        while(time < max_duration):
            ret, frame = cap.read()
            if ret == False:
                break
            time += 1
            # Convert the region of interest to grayscale.
            gray = cv2.cvtColor(frame[self.im_coor[1]:self.im_coor[0],10:-10],
                                cv2.COLOR_BGR2GRAY)
            # Apply binary threshold to eliminate background.
            # This works because the pavement is always in light color and 
            # vehicle tires always in dark color.
            ret, black = cv2.threshold(gray,64,255,cv2.THRESH_BINARY_INV)
            trace.append(np.sum(black, axis=0))
            elev.append(np.sum(black, axis=1))
            black = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Threshold", black) # Visualize the thresholding effect
            cv2.waitKey(1)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        cap.release()
        trace = np.asarray(trace)
        elev = np.asarray(elev)
        # Save data to files.
        np.save(self.SRC+self.VID.replace(".avi", "_trace"), trace)
        np.save(self.SRC+self.VID.replace(".avi", "_elev"), elev)

    # Convert speed and distance estimates from pixel to meter.
    def px2meter(self, px_speed, Xi):
        im_coor = np.asarray(self.im_coor).astype("float")
        re_coor = np.asarray(self.re_coor).astype("float")
        Ai = im_coor[0]
        Ci = im_coor[1]
        Di = im_coor[2]
        Ar = re_coor[0]
        Cr = re_coor[1]
        Dr = re_coor[2]
        # Calculate the cross ratio in image coordinates.
        CR = ((Ai-Ci)*(Xi-Di))/((Xi-Ci)*(Ai-Di))
        # Back-calculate real world location of the object.
        Xr = (Dr-CR*(Dr-Ar)/(Cr-Ar)*Cr) / (1-CR*(Dr-Ar)/(Cr-Ar))
        # Calculate horizontal pixel to meter conversion factor.
        factor = 2.0*(self.d+Xr)*np.tan(self.theta)/self.dim[1]
        return px_speed*factor*self.fps, self.d+Xr

    # Extract dynamics from discovered traces using DBSCAN and convex hull.
    def analyze_trace_convex(self):
        print "Analyzing trace using CONVEX HULL..."
        # Threshold value to find vehicle traces.
        thresh = 750
        trace = np.load(self.SRC+self.VID.replace(".avi", "_trace.npy"))
        trace = trace[1000:-1000]
        mask = trace > thresh
        # Find the horizontal image coordinates of the foreground pixels.
        trace_pixels = np.column_stack(np.where(mask))
        elev = np.load(self.SRC+self.VID.replace(".avi", "_elev.npy"))
        elev = elev[1000:-1000]
        mask = elev > thresh
        # Find the vertical image coordinates of the foreground pixels.
        elev_pixels = np.column_stack(np.where(mask))

        # Plot histograms for debugging purpose.
        if False:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax1.hist(trace.flatten(),bins=50)
            #ax1.plot(trace_pixels[:,0], trace_pixels[:,1],'.')
            ax2 = fig.add_subplot(2,1,2)
            ax2.hist(elev.flatten(),bins=50)
            #ax2.plot(elev_pixels[:,0], elev_pixels[:,1],'.')
            plt.show()

        fig1 = plt.figure()
        ax11 = fig1.add_subplot(2,1,1)
        ax11.plot(trace_pixels[:,0], trace_pixels[:,1],'.')
        ax11.set_ylim([0,self.dim[1]])
        # Apply DBSCAN to separate inter-vehicle pixels.
        dbscan = DBSCAN(eps=10, min_samples=32)  
        dbscan.fit(trace_pixels)
        labels = dbscan.labels_
        labels_unique = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels_unique)))
        ax12 = fig1.add_subplot(2,1,2)
        count = 0
        log = []
        for label, color in zip(labels_unique, colors):
            if label != -1:
                mask = labels == label
                cluster = trace_pixels[mask]
                if len(cluster) > self.dim[1]:
                    count += 1
                    ax12.plot(cluster[:,0], cluster[:,1], '.', color=color)
                    # Apply convex hull to find vehicle speed in unit of pixel/frame.
                    hull = ConvexHull(cluster)
                    for simplex in hull.simplices:
                         ax12.plot(cluster[simplex,0], cluster[simplex,1], '-or')
                    pairs = np.asarray([cluster[simplex] for simplex 
                                        in hull.simplices]).astype(float)
                    velocity = []
                    for pair in pairs:
                        if abs(pair[1,1] - pair[0,1]) > 1.0/10.0*self.dim[1]:
                            velocity.append((pair[1,1]-pair[0,1])/
                                            (pair[1,0]-pair[0,0]))
                    start = np.min(cluster[:,0])
                    end = np.max(cluster[:,0])
                    # Apply DBSCAN again to remove outliers in vertical heat map.
                    data = np.column_stack(np.where(elev[start:end]>thresh))
                    outlier_detector = DBSCAN(eps=4,min_samples=32)
                    outlier_detector.fit(data)
                    scores = outlier_detector.labels_
                    inliers = data[scores!=-1].tolist()
                    inliers.sort(key=lambda x:x[1])
                    inliers = np.asarray(inliers)
                    outliers = data[scores==-1]
                    # Estimate pixel distance of the vehicle.
                    px_distance = self.im_coor[1] + np.average(inliers[-10:,1])
                    fig2 = plt.figure()
                    ax21 = fig2.add_subplot(3,1,1)
                    pixels = np.column_stack(np.where(trace[start:end]>thresh))
                    ax21.plot(pixels[:,0],pixels[:,1],'.')
                    ax21.set_xlim([0,end-start])
                    ax21.set_ylim([0,self.dim[1]])
                    ax22 = fig2.add_subplot(3,1,2)
                    ax22.plot(cluster[:,0], cluster[:,1], '.', color=color)
                    for simplex in hull.simplices:
                        ax22.plot(cluster[simplex,0], cluster[simplex,1], '-or')
                    ax22.set_xlim([start,end])
                    ax22.set_ylim([0,self.dim[1]])
                    ax23 = fig2.add_subplot(3,1,3)
                    ax23.plot(outliers[:,0],outliers[:,1],'r*')
                    ax23.plot(inliers[:,0],inliers[:,1],'g.')
                    ax23.axhline(y=px_distance-self.im_coor[1])
                    ax23.set_xlim([0,end-start])
                    ax23.set_ylim([0,self.im_coor[0]-self.im_coor[1]])
                    # Save plots for debugging purpose.
                    fig2.savefig(self.SRC+self.VID.replace(".avi",'/')+"vehicle%d.png" % count)
                    plt.close(fig2)
                    px_speed = np.median(velocity)
                    # Convert distance and speed estimates from pixel unit to meter.
                    speed, distance = self.px2meter(px_speed, px_distance)
                    log.append([(start+1000.0)/self.fps,
                                (end+1000.0)/self.fps,
                                speed, distance,
                                px_speed, px_distance])
        ax12.set_ylim([0,self.dim[1]])
        print "Found %d vehicles." % count
        # Save data to file.
        np.save(self.SRC+self.VID.replace(".avi", "_convexlog.npy"), log)
        #plt.show()

    # [deprecated] Extract dynamics from discovered traces using 
    # ransac + linear regression. This method will be removed in the future.
    def analyze_trace_ransac(self):
        print "Analyzing trace using RANSAC..."
        trace = np.load(self.SRC+self.VID.replace(".avi", "_trace.npy"))
        trace = trace[1000:6000]
        mask = trace < 12000
        # Find the image coordinates of the foreground pixels.
        trace_pixels = np.column_stack(np.where(mask))
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(trace_pixels[:,0], trace_pixels[:,1],'.')
        ax1.set_ylim([0,self.dim[1]])
        dbscan = DBSCAN(eps=15, min_samples=100)  
        dbscan.fit(trace_pixels)
        labels = dbscan.labels_
        labels_unique = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels_unique)))
        ax2 = fig.add_subplot(2,1,2)
        count = 0
        log = []
        for label, color in zip(labels_unique, colors):
            if label != -1:
                mask = labels == label
                cluster = trace_pixels[mask]
                if len(cluster) > self.dim[1]:
                    count += 1
                    t = cluster[:,0].reshape(-1, 1)
                    x = cluster[:,1].reshape(-1, 1)
                    model_x = linear_model.RANSACRegressor(linear_model.LinearRegression())
                    model_x.fit(x, t)
                    x_ransac = np.arange(self.dim[1]).reshape(-1, 1)
                    t_ransac = model_x.predict(x_ransac).reshape(-1, 1)
                    velocity = ((np.max(x_ransac) - np.min(x_ransac)) / 
                                (np.max(t_ransac) - np.min(t_ransac)))
                    log.append([np.min(t), velocity])
                    ax2.plot(t, x, '.', color=color)
                    ax2.plot(t_ransac, x_ransac)
        ax2.set_ylim([0,self.dim[1]])
        print "Found %d vehicles." % count
        np.save(self.SRC+self.VID.replace(".avi", "_ransaclog.npy"), log)
        plt.show()

    # Visualize the processed dynamics by visually plotting it
    # on the raw video. (Somewhat buggy still. Needs to be fixed soon.)
    def inspect_dynamics(self,method):
        print "Inspecting dynamics..."
        log = np.load(self.SRC+self.VID.replace(".avi", "_"+method+"log.npy"))
        # Plot distribution of distance and speed estimates.
        if False:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            ax1.hist(log[:,2]*2.23694, bins=25)
            ax1.set_xlabel("Speed (mph)")
            ax1.set_ylabel("Count")
            ax1.set_title("Speed Distribution of %d Vehicles" % len(log))
            ax2 = fig.add_subplot(2,1,2)
            ax2.hist(log[:,3], bins=25)
            ax2.set_xlabel("Distance (m)")
            ax2.set_ylabel("Count")
            ax2.set_title("Distance Distribution of %d Vehicles" % len(log))
            plt.show()
        
        # Sacle the visualization video for better viewing quality.
        scale = 2
        cap = cv2.VideoCapture(self.SRC+self.VID)#.replace(".avi", "avi.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.SRC+'demo.avi',fourcc, self.fps,
                              (self.dim[1]*scale,self.dim[0]*scale))
        time = 0.0
        count = 0
        speed = -1
        distance = -1
        begin = np.inf
        end = -np.inf
        # Generate the visualization video.
        while(1):
            ret, frame = cap.read()
            if ret == False:
                break
            if time < 5400:
                time += 1.0
                continue
            img = cv2.resize(frame,None,fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
            elapsed = time/self.fps
            idx = np.argwhere(log[:,0]==elapsed)
            # Increment counting and update estimates if a new vehicle appears.
            if idx.size != 0:
                count += 1 # Vehicle count
                begin = log[idx,0] # Begin time of the passing (sec)
                end = log[idx,1] # End time of the passing (sec)
                speed = log[idx,2]*2.23694 # Estimated speed (mph)
                distance = log[idx, 3] # Estimated distance to the Pi camera (m)
                px_speed = log[idx, 4]*scale # Estimated pixel speed (px/frame)
                px_distance = log[idx, 5]*scale # Estimated pixel distance (px)
                px_pos = 0 # Position of front bumper (px)
            while begin <= elapsed <= end:
                cv2.circle(img, (px_pos, px_distance), 15, (255,0,0), -1) # Front bumper position (px)
                cv2.line(img, (0,px_distance),(self.dim[1]*scale,px_distance),(255,0,0),2*scale) # Tires-ground contact point (px)
                px_pos += px_speed # Update position
                break
            # Output estimates to frame.
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Elapsed: %.2f" % elapsed,(500*scale,20*scale), 
                        font, 0.5*scale, (255,255,255), 1*scale)
            cv2.putText(img, "Count: %d" % count,(500*scale,40*scale), 
                        font, 0.5*scale, (255,255,255), 1*scale)
            cv2.putText(img, "Speed: %.2f" % speed,(500*scale,60*scale), 
                        font, 0.5*scale, (255,255,255), 1*scale)
            cv2.putText(img, "Distance: %.2f" % distance,(500*scale,80*scale), 
                        font, 0.5*scale, (255,255,255), 1*scale)
            out.write(img)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)
            time += 1.0
            if ret == False:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Instantiate a Validation Cone object.
    # Usage:
    #   ValidationCone(data directory, video path, fps, pixel coordinates, 
    #                  physical coordinates, fov, distance from camera to outer lane)
    vc = ValidationCone("test3/", "freeflow_1.avi", 60.0, 
                        [395, 335, 315], [0.00, 3.70, 7.25], 0.4331, 5.18)
    # Find the heat map/heat trace of the video.
    #vc.find_trace()
    # Estimate vehicle counts, distance, and speeds using DBSCAN and convex hull.
    vc.analyze_trace_convex()
    # Visualize the estimation results in form of histogram and video overlay.
    vc.inspect_dynamics("convex")
