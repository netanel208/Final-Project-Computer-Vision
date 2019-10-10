import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
np.set_printoptions(threshold=sys.maxsize)


# =========================================================================================
# ==================================Auxiliary functions====================================
# =========================================================================================


def findIndexsAndValuesInImg2(pos1, pos2, pos1_4, amount: int):
    """
    This function find the appropriate points in image2 between the matching points list
    and return their [x,y] and their index in matching points list.
    :param pos1: list of key points in image 1
    :param pos2: list of key points in image 2 (that match to points in image 1)
    :param pos1_4: list of 1 or 4 random points from pos1
    :param amount: 1 or 4 amount of points that we want to find their index and [x,y] that mention upon
    :return:
    """
    if amount == 4:
        p1 = None
        p2 = None
        p3 = None
        p4 = None
        ip1 = None
        ip2 = None
        ip3 = None
        ip4 = None
        for i in range(pos1.shape[0]):
            if pos1[i][0] == pos1_4[0][0] and pos1[i][1] == pos1_4[0][1]:
                p1 = pos2[i]
                ip1 = i
            elif pos1[i][0] == pos1_4[1][0] and pos1[i][1] == pos1_4[1][1]:
                p2 = pos2[i]
                ip2 = i
            elif pos1[i][0] == pos1_4[2][0] and pos1[i][1] == pos1_4[2][1]:
                p3 = pos2[i]
                ip3 = i
            elif pos1[i][0] == pos1_4[3][0] and pos1[i][1] == pos1_4[3][1]:
                p4 = pos2[i]
                ip4 = i
        real_pos2_val = np.array([p1, p2, p3, p4])
        real_pos2_ind = np.array([ip1, ip2, ip3, ip4])
        return real_pos2_val, real_pos2_ind
    elif amount == 1:
        p1 = None
        ip1 = None
        for i in range(pos1.shape[0]):
            if pos1[i][0] == pos1_4[0][0] and pos1[i][1] == pos1_4[0][1]:
                p1 = pos2[i]
                ip1 = i
        real_pos2_val = p1
        real_pos2_ind = ip1
        return real_pos2_val, real_pos2_ind


def testTheInliers(H, inliers, pos1: np.ndarray, pos2: np.ndarray):
    """
    This function display the best and maximum inliers in image1 and the result -
    the matching points, in image2 .
    :param inliers: list of max inliers index
    :param pos1: key points array in image1 (part 1 of matching points)
    :param pos2: key points array in image2 (part 2 of matching points)
    :return:
    """
    t1 = []
    t2 = []
    for i in inliers:
        t1.append(pos1[i])
        t2.append(pos2[i])
    print("test : t1 = ", np.array(t1))
    print("test : t2 = ", np.array(t2))
    res = []
    for i in inliers:
        res.append(ApplyHomography(np.array([pos1[i]]), H))
    print("test : pred_t2 = ", res)
    ans = np.array(t2) - res


def testTheHtot(Htot, Hpair):
    """
    Hpair should contain 3 homography matrices
    :param Htot:
    :param Hpair:
    :return:
    """
    if len(hpair) == 3:
        t = []
        H0_1 = Hpair[0]
        H1_2 = Hpair[1]
        H2_3 = Hpair[2]
        # i < 2
        H0_2 = np.dot(H1_2, H0_1)
        H1_2 = H1_2
        # i = 2
        H2_2 = np.identity(3)
        # i > 2
        H3_2 = np.linalg.inv(H2_3)
        t.append(H0_2/H0_2[2][2])
        t.append(H1_2/H1_2[2][2])
        t.append(H2_2)
        t.append(H3_2/H3_2[2][2])
        print("test htot : ", t)


def prepareHpair(setOfImage: list):
    res = []
    for i in range(len(setOfImage)-1):
        im1 = setOfImage[i]
        im2 = setOfImage[i+1]
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        pos1, pos2 = matchFeatures(im1, im2)
        h, inliers = ransacHomography(pos1, pos2, 170, 5)
        res.append(h)
        displayMatches(im1, im2, pos1, pos2, inliers)
    print("Hpair = ", res)
    return res


# =========================================================================================
# =====================================Main functions======================================
# =========================================================================================


def matchFeatures(img1: np.ndarray, img2: np.ndarray):
    """
    Find match point between images(can include outliers).
    Use OpenCV Brute Force(BF) matching.
    :param img1
    :param img2
    :return: set of points coordinates in both images
             , pos1 and pos2, of size nx2 that are most likely matched.
    """
    orb = cv2.ORB_create()

    #  Find the key points and their descriptors with the orb detector
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # The matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Create matches of the descriptors, then sort them based on their distances
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
    pos1 = np.array(list_kp1)
    pos2 = np.array(list_kp2)

    # Checking if the matching points between the images in the right position
    # for i in range(len(list_kp1)):
    #     y1, x1 = int(list_kp1[i][0]), int(list_kp1[i][1])  # x is number of row, y is number of column
    #     y2, x2 = int(list_kp2[i][0]), int(list_kp2[i][1])
    #     print("intensities match : img1[x1][y1], img2[x2][y2] ", img1[x1][y1], img2[x2][y2])
    #     print("location of pixel in img1 : ", x1, y1)
    #     print("location of pixel in img2 : ", x2, y2)
    #     plt.imshow(img1)
    #     plt.show()
    #     plt.imshow(img2)
    #     plt.show()

    # Display matches
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:], None, flags=2)
    plt.imshow(img3)
    plt.show()

    return pos1, pos2


def ApplyHomography(pos1: np.ndarray, H12: np.ndarray) -> np.ndarray:
    """
    The function convert 2D point to 3D homogeneous point and transform the point by homography matrix
    then return back from 3D point to 2D point .
    :param pos1: point
    :param H12: homography matrix
    :return:
    """
    pos2 = None
    i = 0
    for point in pos1:
        homog_point = np.array([point[0], point[1], 1])
        nhomog_point = np.dot(H12, homog_point.T)
        x = nhomog_point[0]/nhomog_point[2]
        y = nhomog_point[1]/nhomog_point[2]
        res_point = np.array([x, y])
        pos2 = res_point
        i += 1
    return pos2


def leastSquareHomograpy(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
        Finds the homography matrix, M, that transforms points from p1 to p2.
        returns the homography and the error between the transformed points to their
        destination (matched) points.
        Error = np.sqrt(sum((M.dot(p1)-p2)**2))
        p1: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        p2: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]
        return: (Homography matrix shape:[3,3], Homography error)
        """
    M = np.zeros((2 * p1.shape[0], 9))
    i = 0
    while i < M.shape[0]-1:
        M[i][0] = -p1[i // 2][0]
        M[i][1] = -p1[i // 2][1]
        M[i][2] = -1
        M[i][6] = (p2[i // 2][0] * p1[i // 2][0])
        M[i][7] = (p2[i // 2][0] * p1[i // 2][1])
        M[i][8] = (p2[i // 2][0])
        M[i + 1][3] = -p1[i // 2][0]
        M[i + 1][4] = -p1[i // 2][1]
        M[i + 1][5] = -1
        M[i + 1][6] = (p2[i // 2][1] * p1[i // 2][0])
        M[i + 1][7] = (p2[i // 2][1] * p1[i // 2][1])
        M[i + 1][8] = (p2[i // 2][1])
        i += 2
    U, D, V = np.linalg.svd(M)
    H = V[len(V)-1]
    H = H / H[len(H) - 1]
    H = np.reshape(H, (3, 3))

    # ===============Test the matrix===================
    # H12, mask = cv2.findHomography(p1, p2, method=0)
    # print("opencv = ", H12)
    # print("[p1[1][0], p1[1][1], 1] = ", [p1[1][0], p1[1][1], 1])
    # p_1 = np.dot(H, [p1[1][0], p1[1][1], 1])
    # p_11 = np.dot(H12, [p1[1][0], p1[1][1], 1])
    # print(p_1/p_1[2], p_11/p_11[2])
    # =================================================

    # ===========Other way to solution=================
    # x1, y1 = p1[0][0], p1[0][1]
    # x2, y2 = p1[1][0], p1[1][1]
    # x3, y3 = p1[2][0], p1[2][1]
    # x4, y4 = p1[3][0], p1[3][1]
    # xp1, yp1 = p2[0][0], p2[0][1]
    # xp2, yp2 = p2[1][0], p2[1][1]
    # xp3, yp3 = p2[2][0], p2[2][1]
    # xp4, yp4 = p2[3][0], p2[3][1]
    #
    # A = [
    #     [-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
    #     [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
    #     [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
    #     [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
    #     [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
    #     [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
    #     [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
    #     [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]]
    #
    # U, S, V = np.linalg.svd(A)
    # # m = min(S)
    # # print("min = ", m)
    # # print("S = ", S)
    # # print("V' = ", np.dot(V, V.T))
    #
    # # H = V[:, V.shape[1] - 2]
    # H = V[8]
    # H = np.reshape(H, (3, 3))
    # print("H = ", H)
    #
    # H12, mask = cv2.findHomography(p1, p2)
    # print("opencv H12 = ", H12)
    #
    # print("[p1[0][0], p1[0][1], 1] = ", [p1[0][0], p1[0][1], 1])
    # p_1 = np.dot(H, [p1[0][0], p1[0][1], 1])
    # p_11 = np.dot(H12, [p1[0][0], p1[0][1], 1])
    #
    # print(p_1/p_1[2], p_11/p_11[2])
    # ===================================================

    # # Test the result matrix
    # try:
    #     t_H, mask = cv2.findHomography(p1, p2, method=0)
    #     print("t_H (homography matrix)= ", t_H)
    #     return t_H
    # except Exception as e:
    #     print("exception = ", p1, p2)
    return H


def E(H12: np.ndarray, pos1_1: np.ndarray, pos1: np.ndarray, pos2: np.ndarray, inlierTol: int) -> (int, int):
    """
    The method find the inliers and outliers of matching points for given of 1 point.
    The method use Squared Euclidean Distance to find inliers points :
    E_i = || P' - P ||^2 < inlierTol
    when P' - is the transformed set of 1 point
    and P - is the matching point in image 2
    :param H12: Homography matrix
    :param pos1_1: Set of 1 point in image 1
    :param pos1: Full set of points in image 1
    :param pos2: Full set of points in image 2
    :param inlierTol: Constant value to inlier point approximation
    :return:
    """
    pred_pos2 = ApplyHomography(pos1_1, H12)

    # find the corresponding point in image 2
    real_pos2_val, real_pos2_ind = findIndexsAndValuesInImg2(pos1, pos2, pos1_1, 1)
    # print("real_pos1 = ", pos1_1)
    # print("pred_pos2 = ", pred_pos2)
    # print("real_pos2 = ", real_pos2_val)

    # Find E = || P' - P ||^2
    sed = (np.linalg.norm(pred_pos2-real_pos2_val))**2
    # print("sed = ", sed)

    # Determine if E < inlierTol
    if sed < inlierTol:
        return 1, real_pos2_ind
    else:
        return 0, None


def ransacHomography(pos1: np.ndarray, pos2: np.ndarray, numIter: int, inlierTol: int) -> (np.ndarray, np.ndarray):
    """
     This function fit homography to maximal inliers given point matches
     using the RANSAC algorithm.
    :param pos1: nx2 matrices containing n rows of [x,y] coordinates of src matched points.
    :param pos2: nx2 matrices containing n rows of [x,y] coordinates of dst matched points.
    :param numIter: number of RANSAC iterations to perform.
    :param inlierTol: inlier tolerance threshold.
    :return:
    """
    maxInliers = []
    for i in range(numIter):

        # Choose 4 different random points
        r_p1 = random.choice(pos1)
        r_p2 = random.choice([a for a in pos1 if a[0] != r_p1[0] and a[1] != r_p1[1]])
        r_p3 = random.choice([a for a in pos1 if a[0] != r_p1[0] and a[1] != r_p1[1]
                              and a[0] != r_p2[0] and a[1] != r_p2[1]])
        r_p4 = random.choice([a for a in pos1 if a[0] != r_p1[0] and a[1] != r_p1[1]
                              and a[0] != r_p2[0] and a[1] != r_p2[1]
                              and a[0] != r_p3[0] and a[1] != r_p3[1]])
        rp_4 = np.array([r_p1, r_p2, r_p3, r_p4])
        real_pos2_val, real_pos2_ind = findIndexsAndValuesInImg2(pos1, pos2, rp_4, 4)
        H = leastSquareHomograpy(rp_4, real_pos2_val)

        # Compute E to full pos1 set and bind the inliers
        inliers = []
        for j in pos1:
            dis, match = E(H, np.array([j]), pos1, pos2, inlierTol)
            if dis == 1:
                inliers.append(match)

        # Find the maximum inliers set
        if len(inliers) > len(maxInliers):
            maxInliers = inliers

    # Calculate the finally homography matrix by inliers
    print("maxInliers = ", maxInliers)
    t_pos1 = np.zeros((len(maxInliers), 2))
    t_pos2 = np.zeros((len(maxInliers), 2))
    ind = 0
    for i in maxInliers:
        t_pos1[ind][0], t_pos1[ind][1] = (pos1[i][0], pos1[i][1])
        t_pos2[ind][0], t_pos2[ind][1] = (pos2[i][0], pos2[i][1])
        ind += 1
    H12 = leastSquareHomograpy(t_pos1, t_pos2)
    # H12, mask = cv2.findHomography(t_pos1, t_pos2)

    # return the finally homography matrix and the set of maxInliers
    return H12, maxInliers


def displayMatches(im1: np.ndarray, im2: np.ndarray, pos1: np.ndarray, pos2: np.ndarray, inlind):
    # numpy_horizontal = np.hstack((im1, im2))
    # plt.imshow(numpy_horizontal)
    # plt.show()
    # fig = plt.figure()
    # x_shift = im1.shape[1]
    #
    # # ================Other way to solution=================
    # # # Mark all key points in red points
    # # for i in inlind:
    # #     x1, y1 = pos1[i][0], pos1[i][1]
    # #     x2, y2 = pos2[i][0]+x_shift, pos2[i][1]
    # #     plt.plot(x1, y1, '.y')
    # #     plt.plot(x2, y2, '.y')
    # #     plt.plot([x1, x2], [y1, y2], 'ro-')
    # # plt.imshow(numpy_horizontal, cmap='Greys')
    # # plt.show()
    # # ======================================================
    #
    # # Mark all key points in red points
    # for i in pos1:
    #     x1, y1 = i[0], i[1]
    #     cv2.circle(numpy_horizontal, (int(x1), int(y1)), 4, (255, 0, 0), 1)
    # for j in pos2:
    #     x2, y2 = j[0] + x_shift, j[1]
    #     cv2.circle(numpy_horizontal, (int(x2), int(y2)), 4, (255, 0, 0), 1)
    # cv2.imshow('img', numpy_horizontal)
    # cv2.waitKey(0)

    # Create a new output image that concatenates the two images together
    rows1 = im1.shape[0]
    cols1 = im1.shape[1]
    rows2 = im2.shape[0]
    cols2 = im2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1, :] = np.dstack([im1, im1, im1])
    # Place the next image to the right of it
    out[:rows2, cols1:cols1 + cols2, :] = np.dstack([im2, im2, im2])

    # Mark all key points in red points
    for i in pos1:
        x1, y1 = i[0], i[1]
        cv2.circle(out, (int(x1), int(y1)), 4, (0, 0, 255), 1)
    for j in pos2:
        x2, y2 = j[0] + cols1, j[1]
        cv2.circle(out, (int(x2), int(y2)), 4, (0, 0, 255), 1)

    # Mark all lines between matching points in blue color
    t = None
    for i in range(0, pos1.shape[0]):
        if i not in inlind:
            x1, y1 = pos1[i][0], pos1[i][1]
            x2, y2 = pos2[i][0] + cols1, pos2[i][1]
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    # Mark all lines between inliers matching points in yellow line
    for i in inlind:
        x1, y1 = pos1[i][0], pos1[i][1]
        x2, y2 = pos2[i][0]+cols1, pos2[i][1]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
    cv2.imshow('img', out)
    cv2.waitKey(0)


def accumulateHomographies(Hpair, m: int):
    """
    Start calculate from the middle(m) :
    when i<m -
    Hi,m = ((((Hm-1*Hm-2)*Hm-3)*Hm-4)...)
    when i>m -
    Hi,m = (((((Hm^-1)*(Hm+1^-1))*(Hm+2^-1))*(Hm+3^-1))...)
    :param Hpair:
    :param m:
    :return:
    """
    Htot = []
    if len(Hpair) > 2:
        for i in range(0, len(Hpair)+1):
            if i < m:
                j = m-2
                H_i_m = Hpair[m-1]
                while j >= i:
                    print("in", j)
                    H_i_m = np.dot(H_i_m, Hpair[j])
                    j -= 1
                Htot.append(np.array(H_i_m)/H_i_m[2][2])
            elif i > m:
                j = m+1
                H_i_m = np.linalg.inv(Hpair[m])
                while i > j:
                    H_i_m = np.dot(H_i_m, np.linalg.inv(Hpair[j]))
                    j += 1
                Htot.append(np.array(H_i_m)/H_i_m[2][2])
            elif i == m:
                Htot.append(np.identity(3))
        print("Htot = ", Htot)
    elif len(Hpair) == 2:
        t1 = np.dot(Hpair[1], Hpair[0])
        t2 = np.identity(3)
        t3 = np.dot(np.linalg.inv(Hpair[0]), np.linalg.inv(Hpair[1]))
        Htot.append(t1/t1[2][2])
        Htot.append(t2)
        Htot.append(t3/t3[2][2])
        print("Htot = ", Htot)
    return Htot


def renderPanorama(im, H):
    """

    :param im: array or dict of n grayscale images.
    :param H:  array or dict array of n 3x3 homography matrices transforming the ith image % coordinates to the panorama
     image coordinates.
    :return:
    """
    #compute corners
    corners = []
    for i in range(0, len(im)):
        icorner1 = [0, 0]
        icorner2 = [0, im[i].shape[0]-1]
        icorner3 = [im[i].shape[1]-1, 0]
        icorner4 = [im[i].shape[1]-1, im[i].shape[0]-1]
        print("shape = ", im[i].shape)
        print("icorner1 = ", icorner1)
        print("icorner2 = ", icorner2)
        print("icorner3 = ", icorner3)
        print("icorner4 = ", icorner4)
        # cv2.circle(img, (icorner1[0], icorner1[1]), 8, (0, 0, 255), 1)
        # cv2.circle(img, (icorner2[0], icorner2[1]), 8, (0, 0, 255), 1)
        # cv2.circle(img, (icorner3[0], icorner3[1]), 8, (0, 0, 255), 1)
        # cv2.circle(img, (icorner4[0], icorner4[1]), 8, (0, 0, 255), 1)
        # cv2.imshow('fig', img)
        # cv2.waitKey(0)
        pcorner1 = ApplyHomography(np.array([icorner1]), H[i])
        pcorner2 = ApplyHomography(np.array([icorner2]), H[i])
        pcorner3 = ApplyHomography(np.array([icorner3]), H[i])
        pcorner4 = ApplyHomography(np.array([icorner4]), H[i])
        print("pcorner1 = ", pcorner1)
        print("pcorner2 = ", pcorner2)
        print("pcorner3 = ", pcorner3)
        print("pcorner4 = ", pcorner4)


# ============================================================================
# =================================Main=======================================
# ============================================================================
# img1 = cv2.imread('oxford1.jpg')
# img2 = cv2.imread('oxford2.jpg')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# pos1, pos2 = matchFeatures(img1, img2)
# # t_p1 = np.array([pos1[0], pos1[1], pos1[2], pos1[3], pos1[4], pos1[5], pos1[6]])
# # t_p2 = np.array([pos2[0], pos2[1], pos2[2], pos2[3], pos2[4], pos2[5], pos2[6]])
# # H = leastSquareHomograpy(t_p1, t_p2)
# # H = leastSquareHomograpy(pos1, pos2)
# # count = E(H, t_p1, pos1, pos2, 1)
# # print(count)
# h, inliers = ransacHomography(pos1, pos2, 170, 5)
# testTheInliers(h, inliers, pos1, pos2)  # the result in file - testRANSAC.txt
# displayMatches(img1, img2, pos1, pos2, inliers)
# hpair = prepareHpair([cv2.imread('backyard1.jpg'), cv2.imread('backyard2.jpg'), cv2.imread('backyard3.jpg')])
hpair = prepareHpair([cv2.imread('office1.jpg'), cv2.imread('office2.jpg'), cv2.imread('office3.jpg'),
                      cv2.imread('office4.jpg')])
htot = accumulateHomographies(hpair, 2)
testTheHtot(htot, hpair)
img1 = cv2.imread('office1.jpg')
img2 = cv2.imread('office2.jpg')
img3 = cv2.imread('office3.jpg')
img4 = cv2.imread('office4.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
renderPanorama([img1, img2, img3, img4], htot)
