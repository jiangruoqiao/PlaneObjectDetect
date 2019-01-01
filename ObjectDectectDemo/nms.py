import numpy as np

def nonMaximumSuppression(boxes, overlapThresh):
    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Initialize the picked bounding boxes list
    pick = []

    # Get the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute the area of the bounding boxes
    area = (x2-x1+1) * (y2-y1+1)
    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    idxs = np.argsort(y2)

    # Loop over the indexes list
    while len(idxs) > 0:
        # Get the last index and put it into pick list as the first value to compare
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # Initialize suppression list
        suppress = [last]

        # Loop over all of the indexes in the idxs list
        for pos in range(0, last):
            # Get the current index
            j = idxs[pos]

            # Find the largest (x,y) coordinates for the start of the bounding box and
            # smallest (x,y) coordinates for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # Compute the width and height of the bounding box
            w = max(0, xx2-xx1+1)
            h = max(0, yy2-yy1+1)

            # Compute the ratio of overlap between the computed bounding box and the
            # bounding box in the area list
            overlap = float(w*h)/area[j]

            # If there is sufficient overlap, suppress the current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # Delete all of the indexes in the suppression list
        idxs = np.delete(idxs, suppress)

    # Return the picked bounding boxes
    return boxes[pick]



