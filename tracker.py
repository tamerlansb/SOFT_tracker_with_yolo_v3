import torch
import cv2
from processing import bbox_transform, image_to_net_inp
from pytorch_yolo_v3.darknet import Darknet
from pytorch_yolo_v3.util import load_classes
from utils import postprocessing
from visualize_utils import get_colors
from sort import *


class Tracker():
    def __init__(self, args):
        super(Tracker).__init__()
        self.args = args
        self.mot_tracker = Sort(max_age=0., min_hits=3, iou_threshold=0.5)
        self.CUDA = torch.cuda.is_available()
        self.device = 'cuda' if self.CUDA else 'cpu'
        self.resolution = int(args.size)
        self.object_confindce_threshold = args.confidence
        self.non_maximum_sus_thr =args.nms_thresh
        self.class_confidence = 0.75
        self.classes_to_track = ['person']

        self.num_classes = 80
        self.colors = get_colors()
        self.classes = load_classes('pytorch_yolo_v3/data/coco.names')

        # Set up the neural network
        print("Loading network.....")
        self.detector_network = Darknet('pytorch_yolo_v3/cfg/yolov3.cfg')
        self.detector_network.load_weights('pytorch_yolo_v3/yolov3.weights')
        print("Network successfully loaded")

        self.detector_network.net_info["height"] = str(self.resolution)
        inp_dim = int(self.detector_network.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32
        if self.CUDA:
            self.detector_network.cuda()
        self.detector_network.eval()

    def __procces_and_filter_predict__(self, net_prediction):
        """
        :return: detected bounding boxes [x1,y1,x2,y2] + [score, class prob, class id]
        """
        roi_objs = postprocessing(net_prediction,
                                  num_classes=self.num_classes,
                                  obj_conf_thr=self.object_confindce_threshold,
                                  nms_thr=self.non_maximum_sus_thr)[0]
        if len(roi_objs) > 0:
            roi_obj_ind = [(self.classes[int(obj[-1])] in self.classes_to_track) for obj in roi_objs]
            roi_obj_ind = np.array(roi_obj_ind) * (roi_objs[:, -2].numpy() >= self.class_confidence)
            roi_objs = roi_objs[np.arange(len(roi_objs))[roi_obj_ind]]
        return roi_objs

    def track(self, path_to_video):
        # open video and parse dir name to  save tracked video
        filename = path_to_video
        tracker_video_name = os.path.splitext(os.path.basename(filename))[0]
        dir_name = os.path.dirname(os.path.abspath(filename))

        # init frames capture
        vidcap = cv2.VideoCapture(filename)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # init tracked video
        output_video = cv2.VideoWriter(os.path.join(dir_name, f'{tracker_video_name}_tracked.avi'),
                                       int(fourcc), fps,
                                       (int(width), int(height)))

        frame_count = 0
        while True:
            success, image = vidcap.read()
            if not success:
                break
            frame_count += 1
            if frame_count % 200 == 0:
                print(f'Processed {frame_count} frames')

            # detection
            net_inp, trans = image_to_net_inp(image, self.resolution)
            prediction = self.detector_network(net_inp.to(self.device), self.CUDA)
            roi_objs = self.__procces_and_filter_predict__(prediction)

            image = np.ascontiguousarray(image, dtype=np.uint8)
            if len(roi_objs) > 0:
                # transformation bounding boxes to original coords
                roi_objs[:, :4] = bbox_transform(roi_objs[:, :4], image.shape[1], image.shape[0], *trans[-3:])

                # tracking and draw bounding boxes with ID
                track_bbs_ids = self.mot_tracker.update(roi_objs[:, :5].numpy())
                for i, bbox in enumerate(track_bbs_ids):
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(image, p1, p2, self.colors[int(bbox[-1])], 2)
                    cv2.putText(image, f'ID:{int(bbox[-1])}', p1,
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA
                                )
            # putting to file
            output_video.write(image)

        cv2.destroyAllWindows()
        output_video.release()
        print(f'Tracking complete!')
        print(f'Total {frame_count} frames processed!')
        return

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Human Tracker')

    parser.add_argument("--path-to-video", default="campus4-c0.avi", type=str,
                        help="path to video. Example: ./input_video_to_track.avi ")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help= "Config file",  default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--size", default="416", type=str,
                        help="Input resolution of the Detector network. "
                             "Increase to increase accuracy. Decrease to increase speed")
    return parser.parse_args()


def main_track(args):
    tracker = Tracker(args)

    path_to_video = args.path_to_video
    tracker.track(path_to_video)


if __name__ == '__main__':
    args = arg_parse()

    main_track(args)
