import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="Arial", size=18)
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

class TrafficEstimator:
    """
    Usage:
       TrafficEstimator(data directory, video file name, fps, pixel coordinates, 
                        physical coordinates, fov, camera rotation, 
                        distance from camera to outer lane)
    """
    # Initialize the validation class.
    def __init__(self, SRC, VID, fps, div, im_coor, re_coor, alpha, theta, d):
        self.SRC = SRC # Parent source path
        self.VID = VID # Subpath to video file
        cap = cv2.VideoCapture(self.SRC+self.VID) # Read in source video
        self.dim = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))] # Size of video frame
        #print self.dim
        self.duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
        self.fps = fps # Frames per second
        self.div = div # Vertical division
        self.im_coor = im_coor # Input image coordinates
        self.re_coor = re_coor # Input physical coordinates
        self.alpha = alpha # 1/2 camera field of view
        self.theta = theta # Rotation
        if self.theta != 0:
            self.beta = np.arctan(np.tan(self.alpha)/np.cos(self.theta)) # 1/2 effective field of view
            self.length = self.dim[1]-self.dim[0]/(1.0+1.0/np.tan(self.theta)) # Effective lane width in px
        else:
            self.beta = self.alpha
            self.length = self.dim[1]
        self.d = d # Distance from PiCam to outer lane
        self.kernel = np.ones((3,3),np.uint8)

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
            gray = cv2.cvtColor(frame[self.im_coor[1]:self.im_coor[0],self.div[0]:self.div[1]],
                                cv2.COLOR_BGR2GRAY)
            # Apply binary threshold to eliminate background.
            # This works because the pavement is always in light color and 
            # vehicle tires always in dark color.
            ret, black = cv2.threshold(gray,48,255,cv2.THRESH_BINARY_INV)
            black = cv2.erode(black, self.kernel, iterations=2)
            black = cv2.dilate(black, self.kernel, iterations=2)
            trace.append(np.sum(black, axis=0))
            elev.append(np.sum(black, axis=1))
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            black = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Threshold", np.vstack((gray,black))) # Visualize the thresholding effect
            cv2.waitKey(1)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        cap.release()
        trace = np.asarray(trace)
        elev = np.asarray(elev)
        # Save data to files.
        np.save(self.SRC+self.VID.replace(".mp4", "_trace"), trace)
        np.save(self.SRC+self.VID.replace(".mp4", "_elev"), elev)

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
        factor = 2.0*(self.d+Xr)*np.tan(self.beta)/self.length
        return px_speed*factor*self.fps, self.d+Xr

    # Extract dynamics from discovered traces using DBSCAN and convex hull.
    def analyze_trace(self):
        print "Analyzing trace using CONVEX HULL..."
        # Threshold value to find vehicle traces.
        thresh = 750
        trace = np.load(self.SRC+self.VID.replace(".mp4", "_trace.npy"))
        mask = trace > thresh
        # Find the horizontal image coordinates of the foreground pixels.
        trace_pixels = np.column_stack(np.where(mask))
        elev = np.load(self.SRC+self.VID.replace(".mp4", "_elev.npy"))
        mask = elev > thresh
        # Find the vertical image coordinates of the foreground pixels.
        elev_pixels = np.column_stack(np.where(mask))

        # Plot histograms for debugging purpose.
        if False:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            #ax1.hist(trace.flatten(),bins=50)
            ax1.plot(trace_pixels[:,0], trace_pixels[:,1],'.')
            ax2 = fig.add_subplot(2,1,2)
            #ax2.hist(elev.flatten(),bins=50)
            ax2.plot(elev_pixels[:,0], elev_pixels[:,1],'.')
            plt.show()

        # Apply DBSCAN to separate inter-vehicle pixels.
        dbscan = DBSCAN(eps=10, min_samples=32)  
        dbscan.fit(trace_pixels)
        labels = dbscan.labels_
        labels_unique = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels_unique)))
        count = 0
        log = []
        for label, color in zip(labels_unique, colors):
            if label != -1:
                mask = labels == label
                cluster = trace_pixels[mask]
                if len(cluster) > self.dim[1]:
                    # Apply convex hull to find vehicle speed in unit of pixel/frame.
                    hull = ConvexHull(cluster)
                    pairs = np.asarray([cluster[simplex] for simplex 
                                        in hull.simplices]).astype(float)
                    velocity = []
                    for pair in pairs:
                        if (abs(pair[1,1] - pair[0,1]) > 1.0/20.0*self.dim[1] and
                            pair[1,0]-pair[0,0] != 0):
                            velocity.append((pair[1,1]-pair[0,1])/
                                            (pair[1,0]-pair[0,0]))
                    if len(velocity) != 0:
                        px_speed = np.median(velocity)*1.125
                    else:
                        px_speed = 50
                        print "Unable to estimate velocity!"
                        print np.min(cluster[:,0])
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
                    try:
                        px_distance = self.im_coor[1] + np.average(inliers[-5:,1])
                    except:
                        print inliers
                        px_distance = self.im_coor[1] + 15
                    # Convert distance and speed estimates from pixel unit to meter.
                    speed, distance = self.px2meter(px_speed, px_distance)
                    if px_speed >= 0:
                        count += 1
                        log.append([(start)/self.fps,
                                    (end)/self.fps,
                                    speed, distance,
                                    px_speed, px_distance])
        print "Found %d vehicles." % count
        # Save data to file.
        np.save(self.SRC+self.VID.replace(".mp4", ".npy"), log)

    # Visualize the processed dynamics by visually plotting it
    # on the raw video. (Somewhat buggy still. Needs to be fixed soon.)
    def inspect_dynamics(self):
        print "Inspecting dynamics..."
        log = np.load(self.SRC+self.VID.replace(".mp4", ".npy"))        
        # Sacle the visualization video for better viewing quality.
        scale = 2
        cap = cv2.VideoCapture(self.SRC+self.VID)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.SRC+'demo.avi',fourcc, self.fps,
                              (self.dim[1]*scale,(self.dim[0]-150)*scale))
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
            if time < 0:
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
                px_pos = self.div[0]*scale # Position of front bumper (px)
            while begin <= elapsed <= end:
                cv2.line(img,(px_pos,0),(px_pos,self.dim[0]*scale),(255,0,0),2*scale) # Front bumper position (px)
                cv2.line(img,(0,px_distance),(self.dim[1]*scale,px_distance),(255,0,0),2*scale) # Tires-ground contact point (px)
                px_pos += px_speed # Update position
                break
            # Output estimates to frame.
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(img, "Elapsed: %.2f sec" % elapsed,(10*scale,127*scale), 
                        font, 0.5*scale, (255,255,255), 1*scale)
            cv2.putText(img, "Count: %d veh" % count,(10*scale,147*scale), 
                        font, 0.5*scale, (255,255,255), 1*scale)
            cv2.putText(img, "Speed: %.2f mph" % speed,(10*scale,167*scale), 
                        font, 0.5*scale, (255,255,255), 1*scale)
            cv2.putText(img, "Distance: %.2f m" % distance,(10*scale,187*scale), 
                        font, 0.5*scale, (255,255,255), 1*scale)
            img = img[200:-100,:]
            out.write(img)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)
            time += 1.0
            if ret == False:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()