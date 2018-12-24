# A script for converting OME-TIFF labeled masks to the Particle Tracking Challenge format
# Author: Martin Maska <xmaska@fi.muni.cz>, 2018

import tifffile as tiff

# Convert the tracking results saved in an OME-TIFF image to a dictionary of tracks
def img_to_tracks(fname,X,Y,Z,T):
    img = tiff.TiffFile(fname)
    track_dict = {}
    img_data = img.asarray().ravel()
    for t in range(T):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    index = t*Z*Y*X + z*Y*X + y*X + x 
                    val = img_data[index]
                    if val > 0:
                        if val not in track_dict:
                            track_dict[val] = [[t, x, y, z]]
                        else:
                            track_dict[val].append([t, x, y, z])
    
    return track_dict

# Convert the dictionary of tracks to the XML format used in the Particle Tracking Challenge
def tracks_to_xml(fname, track_dict, keep_labels):
    with open(fname, "w") as f:
        f.write('<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n')
        f.write('<root>\n')
        f.write('<TrackContestISBI2012 SNR=\"1\" density=\"low\" scenario=\"vesicle\">\n')

        for track in track_dict:
            if keep_labels: f.write('<particle>\n')
            for point in track_dict[track]:
                if not keep_labels: f.write('<particle>\n')
                f.write('<detection t=\"'+str(point[0])+'\" x=\"'+str(point[1])+'\" y=\"'+str(point[2])+'\" z=\"'+str(point[3])+'\"/>\n')
                if not keep_labels: f.write('</particle>\n')
            if keep_labels: f.write('</particle>\n')

        f.write('</TrackContestISBI2012>\n')
        f.write('</root>\n')
		
