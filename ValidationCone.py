import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="FreeSans", size=18)
from sklearn.cluster import DBSCAN, MeanShift
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity

from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

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
    def find_wheels(self):
        contact = [] # Ground contact point
        cap = cv2.VideoCapture(self.SRC+self.VID)
        max_duration = self.duration
        time = 0
        while(time < max_duration):
            ret, frame = cap.read()
            if ret == False:
                break
            time += 1
            image_rgb = frame[self.im_coor[1]-40:self.im_coor[0],10:-10]
            ret, thresh = cv2.threshold(image_rgb,96,255,cv2.THRESH_BINARY_INV)
            image_gray = color.rgb2gray(image_rgb)
            edges = canny(image_gray, sigma=2.0,
                        low_threshold=0.0625, high_threshold=0.125)
            edges_rgb = color.gray2rgb(img_as_ubyte(edges))
            if np.sum(thresh) > 2e7:
                hough_radii = np.arange(30, 50)
                hough_res = hough_circle(edges, hough_radii)
                centers = []
                accums = []
                radii = []
                for radius, h in zip(hough_radii, hough_res):
                    # For each radius, extract two circles
                    num_peaks = 2
                    peaks = peak_local_max(h, num_peaks=num_peaks)
                    centers.extend(peaks)
                    accums.extend(h[peaks[:, 0], peaks[:, 1]])
                    radii.extend([radius] * num_peaks)
                grd = []; ctr = []; rad = [];
                # Draw the most prominent 10 circles
                for idx in np.argsort(accums)[::-1][:16]:
                    if accums[idx] > 0.30: # Minimum required Hough response
                        center_x, center_y = centers[idx]
                        radius = radii[idx]
                        cx, cy = circle_perimeter(center_y, center_x, radius)
                        cv2.circle(image_rgb,(center_y,center_x),radius,(0,0,255),1)
                        grd.append(np.max(cy));ctr.append([center_y, center_x]);rad.append(radius);
                # Use kernel density estimation to find wheel centers
                #X = np.asarray(ctr).reshape(-1,1)
                #X_plot = np.arange(0,self.dim[0]).reshape(-1,1)
                #kde = KernelDensity(kernel="gaussian", bandwidth=5.0).fit(X)
                #log_dens = kde.score_samples(X_plot)
                #plt.figure()
                #plt.plot(X_plot, np.exp(log_dens))
                #plt.show()
                
                # Use mean shift to find wheel center(s)
                X = np.asarray(ctr)
                if len(X) != 0:
                    MS = MeanShift(bandwidth=10.0)
                    MS.fit(X)
                    labels = MS.labels_
                    cluster_centers = MS.cluster_centers_
                    labels_unique = np.unique(labels)
                    n_clusters_ = len(labels_unique)
                    pos = []
                    for k in range(n_clusters_):
                        pos.append(cluster_centers[k][1])
                    while len(pos) < 2:
                        pos.append(np.nan)
                else:
                    pos = [np.nan,np.nan]
                contact.append([time,np.median(grd),pos[0],pos[1],np.median(rad)])

            cv2.imshow("Detection", np.vstack((image_rgb, edges_rgb))) # Visualize the process
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        cap.release()
        plt.figure()
        plt.plot()
        
        np.save(self.SRC+self.VID.replace(".avi", "_contact"), contact)

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
        contact = np.load(self.SRC+self.VID.replace(".avi", "_contact.npy"))
        plt.figure()
        plt.plot(contact[:,0], contact[:,2], '.')
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
    vc = ValidationCone("test4/", "050917_V2.avi", 60.0, 
                        [445, 367, 348], [0.00, 3.7, 7.4], 0.443, 4.00)
    # Find the heat map/heat trace of the video.
    #vc.find_wheels()
    # Estimate vehicle counts, distance, and speeds using DBSCAN and convex hull.
    vc.analyze_trace_convex()
    # Visualize the estimation results in form of histogram and video overlay.
    #vc.inspect_dynamics("convex")
