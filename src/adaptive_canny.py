import argparse
import cv2 as cv
import numpy as np
import skimage


class CannyEdgeDetector:
    """Class for adaptive Canny edge detection with interactive trackbars."""
    
    def __init__(
        self, image: np.ndarray,
        window_name: str = "Adaptive Canny Edge Detection",
        camera: cv.VideoCapture | None = None
    ) -> None:
        """Initialize the edge detector.
        
        Args:
            image: Source image (grayscale)
            window_name: Name of the OpenCV window
            camera: VideoCapture object for live camera feed (optional)
        """
        self._img_original = image
        self._window_name = window_name
        self._min_threshold = 0
        self._max_threshold = 0
        self._camera = camera
        
    def _get_edges(self, img: np.ndarray, cmin: int, cmax: int) -> np.ndarray | None:
        """Blur the image and detect its edges.
        
        Args:
            img: Source image
            cmin: Minimum threshold for Canny
            cmax: Maximum threshold for Canny
            
        Returns:
            Processed image with detected edges or None on error
        """
        try:
            img_blurred = cv.GaussianBlur(img, (5, 5), 1)
            edges = cv.Canny(img_blurred, cmin, cmax)
        except Exception as e:
            print(f"get_edges(): Error processing image - {e}")
            return None
        return edges
    
    def _update_canny(self, val: int) -> None:
        """Callback function for trackbars - updates the Canny edge detection result.
        
        Args:
            val: Trackbar value (unused but required by OpenCV callback)
        """
        try:
            self._min_threshold = cv.getTrackbarPos('Min Threshold', self._window_name)
            self._max_threshold = cv.getTrackbarPos('Max Threshold', self._window_name)
            
            if self._min_threshold > self._max_threshold:
                self._min_threshold = self._max_threshold
                cv.setTrackbarPos('Min Threshold', self._window_name, self._min_threshold)
            
            if self._camera is not None and self._camera.isOpened():
                ret, frame = self._camera.read()
                if ret:
                    self._img_original = frame
            
            if len(self._img_original.shape) == 3:
                img = cv.cvtColor(self._img_original, cv.COLOR_BGR2GRAY)
            else:
                img = self._img_original.copy()

            edges = self._get_edges(img, self._min_threshold, self._max_threshold)
            
            if edges is None:
                return
            
            h, w = img.shape
            edges_resized = cv.resize(edges, (w, h))
            
            combined = np.hstack((img, edges_resized))
            
            text = f"Min: {self._min_threshold}, Max: {self._max_threshold}"
            cv.putText(combined, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            cv.putText(combined, "Original", (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            cv.putText(combined, "Edges", (w + 10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv.imshow(self._window_name, combined)
        except Exception as e:
            print(f"_update_canny(): Error updating display - {e}")
    
    def run(self):
        """Run the interactive edge detection application."""
        try:
            cv.namedWindow(self._window_name)
            
            cv.createTrackbar('Min Threshold', self._window_name, 
                             self._min_threshold, 500, self._update_canny)
            cv.createTrackbar('Max Threshold', self._window_name, 
                             self._max_threshold, 500, self._update_canny)
            
            self._update_canny(0)
            
            self._print_instructions()
            
            while True:
                if self._camera is not None:
                    self._update_canny(0)
                
                key = cv.waitKey(1) & 0xFF
                
                if key == 27:
                    break
            
            self._print_final_values()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"run(): Error during execution - {e}")
        finally:
            if self._camera is not None:
                self._camera.release()
            cv.destroyAllWindows()
    
    def _print_instructions(self):
        """Print usage instructions to console."""
        print("Use trackbars to adjust thresholds:")
        print("  - Min Threshold: Lower edge detection threshold")
        print("  - Max Threshold: Upper edge detection threshold")
        print("\nRecommended values:")
        print("  - For coins image: Min=50, Max=150")
        print("  - Experiment to find optimal values!")
        print("\nPress ESC to quit")
        print(f"Current values - Min: {self._min_threshold}, Max: {self._max_threshold}")
    
    def _print_final_values(self):
        """Print final threshold values to console."""
        print("\n=== Final Optimal Values ===")
        print(f"Min Threshold: {self._min_threshold}")
        print(f"Max Threshold: {self._max_threshold}")


def load_image(
        use_camera: bool = False,
        image_path: str | None = None
) -> tuple[np.ndarray | None, cv.VideoCapture | None]:
    """Load image from camera, file path, or use test image.
    
    Args:
        use_camera: If True, capture from camera
        image_path: Path to image file; if None, use test image
        
    Returns:
        Tuple of (image, camera_object) or (None, None) on error.
        Camera object is returned only when using camera, otherwise None.
    """
    try:
        if use_camera:
            cam = cv.VideoCapture(0)
            ret, img = cam.read()
            
            if not ret:
                print("Error: camera capture unsuccessful")
                cam.release()
                return None, None
            
            return img, cam
        elif image_path:
            img = cv.imread(image_path)
            
            if img is None:
                print(f"Error: Could not load image from '{image_path}'")
                print("Please check that the file exists and is a valid image format")
                return None, None
            
            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            return img, None
        else:
            return skimage.data.coins(), None
    except Exception as e:
        print(f"load_image(): Error loading image - {e}")
        return None, None


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Adaptive Canny Edge Detection',
        epilog='Examples:\n'
               '  python3 adaptive_canny.py                    # Use test image (coins)\n'
               '  python3 adaptive_canny.py -c                 # Use camera\n'
               '  python3 adaptive_canny.py -i image.jpg       # Load from file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", dest="use_camera", 
                       help="Use the camera input instead of the test image", 
                       action="store_true")
    parser.add_argument("-i", "--image", dest="image_path",
                       help="Path to the input image file",
                       type=str, default=None)
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        
        if args.use_camera and args.image_path:
            print("Error: Cannot use both camera (-c) and image file (-i) at the same time")
            print("Please choose one option.")
            return
        
        img, camera = load_image(use_camera=args.use_camera, image_path=args.image_path)
        if img is None:
            return
        
        detector = CannyEdgeDetector(img, camera=camera)
        detector.run()
    except Exception as e:
        print(f"main(): Unexpected error - {e}")


if __name__ == "__main__":
    main()
