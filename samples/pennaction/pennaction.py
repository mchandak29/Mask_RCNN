"""
Mask R-CNN
Train on the PennAction dataset and find golf, baseball bat and tennis racket.

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug
import matplotlib.pyplot as plt
import pickle
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

label_map = {"baseball_bat":1,"Golf Stick":2,"tennis_racket":3}

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class PennActionConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pennaction"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 4  # Background + tennis_racket + baseball_bat + golf_club

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    
    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 4

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class PennActionDataset(utils.Dataset):
    
    def load_pennaction(self, dataset_dir, subset):
        """Load a subset of the PennAction dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("pennaction", 1, "baseball_bat")  #source,class_id,name
        self.add_class("pennaction", 2, "Golf Stick")
        self.add_class("pennaction", 3, "tennis_racket")
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        for i,_,c in os.walk(dataset_dir):
            for n in c:
                if n[-4:]=='.jpg':
                    fname = i+'/'+n[:-4]

                    if os.path.isfile(fname+'.json'):
                        annotations = json.load(open(fname+'.json'))
                        height = annotations['imageHeight']
                        width = annotations['imageWidth']
                        label = annotations['shapes'][0]['label']
                        polygons = [a['points'] for a in annotations['shape']]
                            
                        self.add_image(
                            "pennaction",
                            image_id=fname,  # use file name as a unique image id
                            path=fname+'.jpg',
                            width=width, height=height,
                            label = label,
                            polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "pennaction":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        masks = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon([a[1] for a in p], [a[0] for a in p])
            masks[rr, cc, i] = 1
        
        mask = np.max(masks,axis=2)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), [label_map[info['label']]]

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pennaction":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training datasets.
    dataset_train = PennActionDataset()
    dataset_train.load_pennaction(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PennActionDataset()
    dataset_val.load_pennaction(args.dataset, "val")
    dataset_val.prepare()
    
#     augmentation = imgaug.augmenters.Fliplr(0.5)
    augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads',
                augmentation=augmentation)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):

#     with open('/home/code-base/dataset/files.pickle', 'rb') as handle:
#         fi = pickle.load(handle)
    
#     for fname in tqdm(fi):
#         image = skimage.io.imread(fname)
#         results = model.detect([image], verbose=0)
#         # Visualize results
#         r = results[0]
#         r['csr'] = []
#         for j in range(r['masks'].shape[2]):
#             r['csr'].append(csr_matrix(r['masks'][:,:,j]))
#         r.pop('masks', None)

#         with open(fname[:-4]+".pickle", 'wb') as handle:
#             pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if not(image_path or video_path):
    
        with open('/home/code-base/dataset/all_files.pickle','rb') as handle:
            files = pickle.load(handle)
        
        for fname in tqdm(files):
            if (os.path.getmtime(fname[:-4]+'.pickle')<1561100000):
                image = skimage.io.imread(fname)
                results = model.detect([image], verbose=0)
                r = results[0]
                r['csr'] = []
                for j in range(r['masks'].shape[2]):
                    r['csr'].append(csr_matrix(r['masks'][:,:,j]))
                r.pop('masks', None)

                with open(fname[:-4]+".pickle", 'wb') as handle:
                    pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)
                 
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        r['csr'] = []
        for j in range(r['masks'].shape[2]):
            r['csr'].append(csr_matrix(r['masks'][:,:,j]))
        r.pop('masks', None)

        with open(image_path[-8:-4]+".pickle", 'wb') as handle:
            pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Color splash
        #print(r)
#         visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                                     ['BG','baseball_bat','Golf Stick','tennis_racket'], r['scores'])
        #splash = color_splash(image, r['masks'])
        # Save output
        #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        #skimage.io.imsave(file_name, splash)
        
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    #print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect our objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/pennaction/dataset/",
                        help='Directory of the PennAction dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
#     elif args.command == "splash":
#         assert args.image or args.video,\
#                "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PennActionConfig()
    else:
        class InferenceConfig(PennActionConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
