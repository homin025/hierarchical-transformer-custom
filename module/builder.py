import torch
from tensorboardX import SummaryWriter

from module.optimizer import Optimizer

from module.beam import GNMTGlobalScorer
from module.translator import Translator

from module.loss import build_loss_compute
from module.trainer import Trainer, _tally_parameters

from util.report_manager import ReportMgr
from util.logging import logger


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    optim = Optimizer(
        args.optim, args.lr, args.max_grad_norm,
        beta1=args.beta1, beta2=args.beta2,
        decay_method=args.decay_method,
        warmup_steps=args.warmup_steps, model_size=args.enc_hidden_size)

    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        optim.optimizer.load_state_dict(checkpoint['optim'])
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if optim.method == 'adam' and len(optim.optimizer.state) < 1:
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


def build_translator(args, model, tokenizer, symbols, logger=None):
    scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')
    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


def build_trainer(args, device_id, model, symbols, vocab_size, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    train_loss = build_loss_compute(
        model.generator, symbols, vocab_size, device, train=True, label_smoothing=args.label_smoothing)
    valid_loss = build_loss_compute(
        model.generator, symbols, vocab_size, train=False, device=device)

    shard_size = args.max_generator_batches
    grad_accum_count = args.accum_count
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, train_loss, valid_loss, optim, shard_size, grad_accum_count, n_gpu, gpu_rank,
                      report_manager)

    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)

    return trainer
