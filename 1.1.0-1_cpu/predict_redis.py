from datetime import datetime
import io
import numpy as np
import traceback
import cv2

from mmseg.apis import inference_model, init_model
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()

        array = np.frombuffer(msg_cont.message['data'], np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)

        prediction = inference_model(config.model, image)
        pr_mask = prediction.pred_sem_seg
        pr_mask = np.array(pr_mask.cpu().values()[0], dtype=np.uint8)
        pr_mask = np.transpose(pr_mask, (1, 2, 0))

        # not grayscale?
        if config.prediction_format == "bluechannel":
            pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)
            pr_mask[:, :, 1] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])
            pr_mask[:, :, 2] = np.zeros([pr_mask.shape[0], pr_mask.shape[1]])

        out_data = cv2.imencode('.png', pr_mask)[1].tobytes()
        msg_cont.params.redis.publish(msg_cont.params.channel_out, out_data)

        if config.verbose:
            log("process_images - prediction image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)

    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


if __name__ == '__main__':
    parser = create_parser('MMSegmentation - Prediction (Redis)', prog="mmseg_predict_redis", prefix="redis_")
    parser.add_argument('--model', help='Path to the trained model checkpoint', required=True, default=None)
    parser.add_argument('--config', help='Path to the config file', required=True, default=None)
    parser.add_argument('--prediction_format', metavar='FORMAT', default="grayscale", choices=["grayscale", "bluechannel"], help='The format for the prediction images')
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parsed = parser.parse_args()

    try:
        model = init_model(parsed.config, parsed.model, device="cpu")

        config = Container()
        config.model = model
        config.prediction_format = parsed.prediction_format
        config.verbose = parsed.verbose

        params = configure_redis(parsed, config=config)
        run_harness(params, process_image)

    except Exception as e:
        print(traceback.format_exc())
