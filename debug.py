import cv2
import numpy as np


def plot_correspondences(
    im1,
    im2,
    pts1,
    pts2,
    radius=5,
    circle_thickness=2,
    line_thickness=1,
    scale=1.0,
    max_points=None,
    window_name="Correspondences",
    seed=0
):
    """
    Plot colored correspondences between two images.

    Fixes RGB/BGR visualization issues by converting
    RGB images to OpenCV's BGR format automatically.
    """

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError("pts1 and pts2 must have same length")

    if max_points is not None:
        pts1 = pts1[:max_points]
        pts2 = pts2[:max_points]

    def prepare_image(im):
        """
        Convert image to proper OpenCV display format.
        """

        # Grayscale
        if len(im.shape) == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        # RGBA -> BGR
        if im.shape[2] == 4:
            return cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)

        # RGB -> BGR
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    im1_vis = prepare_image(im1)
    im2_vis = prepare_image(im2)

    h1, w1 = im1_vis.shape[:2]
    h2, w2 = im2_vis.shape[:2]

    # Side-by-side canvas
    canvas = np.zeros(
        (max(h1, h2), w1 + w2, 3),
        dtype=np.uint8
    )

    canvas[:h1, :w1] = im1_vis
    canvas[:h2, w1:w1+w2] = im2_vis

    rng = np.random.default_rng(seed)

    for (x1, y1), (x2, y2) in zip(pts1, pts2):

        color = tuple(
            int(c) for c in rng.integers(0, 256, size=3)
        )

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)) + w1, int(round(y2)))

        # Hollow circles
        cv2.circle(
            canvas,
            p1,
            radius,
            color,
            circle_thickness
        )

        cv2.circle(
            canvas,
            p2,
            radius,
            color,
            circle_thickness
        )

        # Matching line
        cv2.line(
            canvas,
            p1,
            p2,
            color,
            line_thickness
        )

    if scale != 1.0:
        canvas = cv2.resize(
            canvas,
            None,
            fx=scale,
            fy=scale
        )

    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return canvas