from TrafficEstimator import TrafficEstimator

vc = TrafficEstimator("./", "sample.mp4", 60.0, [10,-10],
                      [400, 345, 332], [0.00, 3.66, 7.32], 0.4669, 0.0, 4.27)
# Find the heat map/heat trace of the video.
vc.find_trace()
# Estimate vehicle counts, distance, and speeds using DBSCAN and convex hull.
vc.analyze_trace()
# Visualize the estimation results in form of histogram and video overlay.
vc.inspect_dynamics()
