import os
import datetime

from utils.meters import set_summary_writer
from utils.distributed import master_only_print as print
from utils.distributed import master_only


def get_date_uid():
    """Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '20171122171307111552'.
    """
    return str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))


def init_logging(opt):
    date_uid = get_date_uid()
    if opt.name is not None:
        logdir = os.path.join(opt.checkpoints_dir, opt.name)
    else:
        logdir = os.path.join(opt.checkpoints_dir, date_uid)
    opt.logdir = logdir
    return date_uid, logdir
 
@master_only
def make_logging_dir(logdir, date_uid):
    r"""Create the logging directory

    Args:
        logdir (str): Log directory name
    """

    
    print('Make folder {}'.format(logdir))
    os.makedirs(logdir, exist_ok=True)
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    image_dir = os.path.join(logdir, 'image')
    eval_dir = os.path.join(logdir, 'evaluation')
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    set_summary_writer(tensorboard_dir)
    loss_log_name = os.path.join(logdir, 'loss_log.txt')
    with open(loss_log_name, "a") as log_file:
        log_file.write('================ Training Loss (%s) ================\n' % date_uid)
