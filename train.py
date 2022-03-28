from turtle import update
import torch
import argparse
from tqdm import tqdm
import time
import os
import sys
import argparse

from utils.utils import logging, logging_csv

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)

def set_up_logging():
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + args.log + '/'
    
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    
    logging = utils.logging(log_path + 'log.txt')
    logging_csv = utils.logging_csv(log_path + 'record.csv')
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('-beam_search', default=False, action='store_true', help='beam_search')
    parser.add_argument('-config', default='config.yaml', type=str, help='config file')
    parser.add_argument('-model', default='graph2seq', type=str, choice=['seq2seq', 'graph2seq', 'bow2seq', 'h_attention'])
    parser.add_argument('-gpus', default=[1], type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore',
                        type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=1234,
                        help="Random seed")
    parser.add_argument('-notrain', default=False, action='store_true',
                        help="train or not")
    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-verbose', default=False, action='store_true',
                        help="verbose")
    parser.add_argument('-adj', type=str, default="numsent",
                        help='adjacent matrix')
    parser.add_argument('-use_copy', default=False, action="store_true",
                        help='whether to use copy mechanism')
    parser.add_argument('-use_bert', default=False, action="store_true",
                        help='whether to use bert in the encoder')
    parser.add_argument('-use_content', default=False, action="store_true",
                        help='whether to use title in the seq2seq')
    parser.add_argument('-word_level_model', default='bert', choices=['bert', 'memory', 'word'],
                        help='whether to use bert or memory network or nothing in the word level of encoder')
    parser.add_argument('-graph_model', default='none', choices=['GCN', 'GNN', 'none'],
                        help='whether to use gcn in the encoder')
    parser.add_argument('-debug', default=False, action="store_true",
                        help='whether to use debug mode')

    opt = parser.parse_args()
    config = utils.utils.read_config(opt.config)
    return opt, config


logging, logging_csv, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()


def train(model, vocab, dataloader, scheduler, optim, updates):
    scores = []
    max_bleu = 0.
    for epoch in range(1, config.epoch + 1):
        total_acc = 0.
        total_loss = 0.
        start_time = time.time()

        if config.schedule:
            scheduler.step()
            print("Decaying learning rate to %g" % scheduler.get_lr()[0])

        model.train()

        train_data = dataloader.train_batches
        for batch in tqdm(train_data, disable=not args.verbose):
            model.zero_grad()
            outputs = model(batch, use_cuda)
            target = batch.tgt
            if use_cuda:
                target = target.cuda()
            
            loss, acc = model.compute_loss(outputs.transpose(0, 1), target.transpose(0, 1)[1:])
            loss.backward()
            total_loss += loss.data.item()
            total_acc += acc

            optim.step()
            updates += 1

            if updates % config.eval_interval == 0 or args.debug:
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train acc: %.3f\n"
                        % (time.time() - start_time, epoch, updates, total_loss / config.eval_interval,
                           total_acc / config.eval_interval))

                print("evaluating after %d updates...\r"%updates)

                score = eval(model, vocab, dataloader, epoch, updates)
                scores.append(score)


                model.train()
                total_loss = 0.
                total_acc = 0.
                start_time = time.time()
                report_total = 0
            

            if updates % config.save_interval == 0:
                save_model(log_path + str(updates) + '_updates_chechpoint.pt', model, optim, updates)

    return max_bleu

def eval(model, vocab, dataloader, epoch, updates, do_test=False):
    model.eval()
    multi_ref, reference, candiate, source, tags, alignments = [], [], [], [], [], []
    if do_test:
        data_batches = dataloader.test_batches
    else:
        data_batches = dataloader.dev_batches
    i = 0
    for batch in tqdm(data_batches, disable=not args.verbose):
        sample, alignment = model.beam_sample(batch, use_cuda, beam_size=config.beam_size)
        candidate += [vocab.id2sent(s) for s in samples]
        source += [example for example in batch.examples]
        multi_ref += [example.org_targets for example in batch.examples]

    text_result, bleu = utils.eval_multi_bleu(multi_ref, candidate, log_path)
    return belu




def save_model(path, model, optim, updates):
    model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates
    }
    torch.save(checkpoints, path)

