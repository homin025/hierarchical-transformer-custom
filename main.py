import os
import random
import signal
import argparse

import torch
import sentencepiece

from module import hierarchical_model, builder, data_loader
from util import distributed
from util.logging import logger, init_logger

model_flags = ['emb_size', 'enc_hidden_size', 'dec_hidden_size', 'enc_layers', 'dec_layers', 'block_size', 'heads',
               'ff_size', 'hier', 'inter_layers', 'inter_heads', 'block_size']


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def multi_main(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i

        procs.append(mp.Process(target=run, args=(args, device_id, error_queue), daemon=True))
        procs[i].start()
        logger.info("Starting process pid: %d" % procs[i].pid)
        error_handler.add_child(procs[i].pid)

    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        train(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


def main(args):
    if args.mode == 'train':
        train(args, 0)
    else:
        step = int(args.test_from.split('.')[-2].split('_')[-1])
        test(args, args.device_id, step)


def train(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    logger.info('Device %s : %d' % (device, device_id))

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in model_flags:
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<pad>')
    symbols = {'BOS': spm.PieceToId('<s>'), 'EOS': spm.PieceToId('</s>'), 'PAD': word_padding_idx,
               'SUM': spm.PieceToId('<sum>'), 'BOP': spm.PieceToId('<p>'), 'EOP': spm.PieceToId('</p>'),
               'SEP': spm.PieceToId('<sep>'), 'MASK': spm.PieceToId('<msk>')}

    vocab = spm
    vocab_size = len(spm)

    model = hierarchical_model.Model(args, word_padding_idx, vocab_size, device, checkpoint)
    optim = builder.build_optim(args, model, checkpoint)
    logger.info(model)

    def train_iter_fct():
        return data_loader.CustomDataLoader(args, data_loader.load_dataset(args, vocab, 'train', shuffle=True), vocab,
                                            symbols, args.batch_size, device, shuffle=True, is_test=False)

    trainer = builder.build_trainer(args, device_id, model, symbols, vocab_size, optim)
    trainer.train(train_iter_fct, args.train_steps)


def test(args, device_id, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    assert args.test_from != '', 'Load model argument is missing'

    logger.info('Loading checkpoint from %s' % args.test_from)
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            setattr(args, k, opt[k])

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<pad>')
    symbols = {'BOS': spm.PieceToId('<s>'), 'EOS': spm.PieceToId('</s>'), 'PAD': word_padding_idx,
               'SUM': spm.PieceToId('<sum>'), 'BOP': spm.PieceToId('<p>'), 'EOP': spm.PieceToId('</p>'),
               'SEP': spm.PieceToId('<sep>'), 'MASK': spm.PieceToId('<msk>')}

    vocab = spm
    vocab_size = len(spm)

    model = hierarchical_model.Model(args, word_padding_idx, vocab_size, device, checkpoint)

    test_iter = data_loader.CustomDataLoader(args, data_loader.load_dataset(args, vocab, 'test', shuffle=False), vocab,
                                             symbols, args.batch_size, device, shuffle=False, is_test=True)

    translator = builder.build_translator(args, model, vocab, symbols, logger=logger)
    translator.translate(test_iter, step)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_file', default='', type=str)
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-data_path', default='./data/', type=str)
    parser.add_argument('-model_path', default='./models/', type=str)
    parser.add_argument('-vocab_path', default='./tokenizer/', type=str)
    parser.add_argument('-train_from', default='', type=str)

    parser.add_argument('-trunc_src_ntoken', default=500, type=int)
    parser.add_argument('-trunc_tgt_ntoken', default=200, type=int)

    parser.add_argument('-emb_size', default=256, type=int)
    parser.add_argument('-enc_layers', default=8, type=int)
    parser.add_argument('-dec_layers', default=1, type=int)
    parser.add_argument('-enc_dropout', default=1, type=float)
    parser.add_argument('-dec_dropout', default=0, type=float)
    parser.add_argument('-enc_hidden_size', default=256, type=int)
    parser.add_argument('-dec_hidden_size', default=256, type=int)
    parser.add_argument('-heads', default=8, type=int)
    parser.add_argument('-ff_size', default=1024, type=int)
    parser.add_argument("-hier", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument('-batch_size', default=10000, type=int)
    parser.add_argument('-valid_batch_size', default=10000, type=int)
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-lr', default=3, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-seed', default=0, type=int)

    parser.add_argument('-train_steps', default=20, type=int)
    parser.add_argument('-save_checkpoint_steps', default=20, type=int)
    parser.add_argument('-report_every', default=100, type=int)

    # multi-gpu
    parser.add_argument('-accum_count', default=1, type=int)
    parser.add_argument('-world_size', default=1, type=int)
    parser.add_argument('-gpu_ranks', default='0', type=str)

    # don't need to change flags
    parser.add_argument("-share_embeddings", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-share_decoder_embeddings", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-max_generator_batches', default=32, type=int)

    # flags for  testing
    parser.add_argument("-test_all", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('-test_from', default='./results', type=str)
    parser.add_argument('-result_path', default='./results', type=str)
    parser.add_argument('-alpha', default=0, type=float)
    parser.add_argument('-length_penalty', default='wu', type=str)
    parser.add_argument('-beam_size', default=5, type=int)
    parser.add_argument('-n_best', default=1, type=int)
    parser.add_argument('-max_length', default=250, type=int)
    parser.add_argument('-min_length', default=20, type=int)
    parser.add_argument("-report_rouge", type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('-dataset', default='', type=str)
    parser.add_argument('-max_wiki', default=5, type=int)

    # flags for  hier
    parser.add_argument('-inter_heads', default=8, type=int)
    parser.add_argument('-inter_layers', default='0', type=str)
    parser.add_argument('-trunc_src_nblock', default=24, type=int)

    # flags for  learning
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=8000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-label_smoothing', default=0.1, type=float)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    args.inter_layers = [int(i) for i in args.inter_layers.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    if args.world_size > 1:
        multi_main(args)
    else:
        main(args)
