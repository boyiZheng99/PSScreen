"""
Configuration file!
"""

import logging
import warnings
import argparse

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# =============================================================================

# ClassNum of Dataset
# =============================================================================
_ClassNum = {
             'retinal_disease': 11
             }


# =============================================================================


# Argument Parse
# =============================================================================
def str2bool(input):
    if isinstance(input, bool):
        return input

    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def show_args(args):
    logger.info("==========================================")
    logger.info("==========       CONFIG      =============")
    logger.info("==========================================")

    for arg, content in args.__dict__.items():
        logger.info("{}: {}".format(arg, content))

    logger.info("==========================================")
    logger.info("===========        END        ============")
    logger.info("==========================================")

    logger.info("\n")


def arg_parse():

    parser = argparse.ArgumentParser(description=' Multi-label Image Recognition with Partial Labels for retinal image analysis')

    # Basic Augments
    parser.add_argument('--post', type=str, default='test', help='postname of save model')
    parser.add_argument('--printFreq', type=int, default='30', help='number of print frequency (default: 1000)')

    parser.add_argument('--dataset', type=str, default='retinal_disease',
                        choices=['retinal_disease'],
                        help='dataset for training and testing')

    parser.add_argument('--pretrainedModel', type=str, default='None', help='path to pretrained model (default: None)')
    parser.add_argument('--resumeModel', type=str, default='None', help='path to resume model (default: None)')
    parser.add_argument('--evaluate', type=str2bool, default='False', help='whether to evaluate model (default: False)')

    parser.add_argument('--epochs', type=int, default=20, help='number of total epochs to run (default: 20)')
    parser.add_argument('--startEpoch', type=int, default=0, help='manual epoch number (default: 0)')
    parser.add_argument('--stepEpoch', type=int, default=10, help='decend the lr in epoch number (default: 10)')

    parser.add_argument('--batchSize', type=int, default=8, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weightDecay', type=float, default=1e-4, help='weight decay (default: 0.0001)')

    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')

    parser.add_argument('--input_size', type=int, default=512, help='training/eval image size')


    parser.add_argument('--gen_psl_epoch', type=int, default=5, help='when to generate pseudo labels(default: 5)')

    parser.add_argument('--lam', type=float, default=1.0, help='pseudo label loss weight')
    parser.add_argument('--alpha', type=float, default=0.95,
                        help='pseudo supervision loss weight')



    args = parser.parse_args()
    args.classNum = _ClassNum[args.dataset]

    return args
# =============================================================================
