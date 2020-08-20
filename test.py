from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime
from tqdm import tqdm

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetuning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--timestamp', type=str, default='',
                    help='Specific training experiment to point to when testing (subfolder of save_folder).')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

args = parser.parse_args()

# Load files from the save folder:
if args.save_folder:
    save_folder = '{}/{}/'.format(args.save_folder, args.timestamp)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')
    log_file = os.path.join(save_folder, 'test_log_static.txt')
    log = open(log_file, 'w')
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args, file=log)
print("CUDA available? {}".format(torch.cuda.is_available()), file=log)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.", file=log)

log.flush()

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(args.batch_size, args.suffix)
# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'rnn':
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)


if args.prior:
    prior_array = []
    prior_array.append(0.9)
    for k in range(1,args.edge_types):
        prior_array.append(0.1/(args.edge_types-1))
    prior = np.array(prior_array)
    print("Using sparsity prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def test():
    acc_test = []
    nll_test = []
    kl_test = []
    mse_test = []
    tot_mse = 0
    counter = 0
    predicted_outputs = []
    actual_outputs = []
    logits_list = []
    edges_list = []

    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))

    with torch.no_grad():
        for batch_idx, (data, relations) in tqdm(enumerate(test_loader)):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()

            assert (data.size(2) - args.timesteps) >= args.timesteps

            data_encoder = data[:, :, :args.timesteps, :].contiguous()
            data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            logits = encoder(data_encoder, rel_rec, rel_send)
    #         edges = gumbel_softmax(logits, tau=args.temp, hard=True)
            edges = torch.nn.functional.one_hot(torch.argmax(logits, axis=-1))  # temporary test -- take out the sampling & use argmax instead
            prob = my_softmax(logits, -1)
            output = decoder(data_decoder, edges, rel_rec, rel_send, 1)
            target = data_decoder[:, :, 1:, :]

            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

            acc = edge_accuracy(logits, relations)
            acc_test.append(acc)

    #         mse_test.append(F.mse_loss(output, target).data[0]) # use with Pytorch 0.2
    #         nll_test.append(loss_nll.data[0])
    #         kl_test.append(loss_kl.data[0])

            mse_test.append(F.mse_loss(output, target).data.item()) # use with new Pytorch
            nll_test.append(loss_nll.data.item())
            kl_test.append(loss_kl.data.item())
            
            # For plotting purposes
            if args.decoder == 'rnn':
                if args.dynamic_graph:
                    output = decoder(data, edges, rel_rec, rel_send, 100,
                                     burn_in=True, burn_in_steps=args.timesteps,
                                     dynamic_graph=True, encoder=encoder,
                                     temp=args.temp)
                else:
                    output = decoder(data, edges, rel_rec, rel_send, 100,
                                     burn_in=True, burn_in_steps=args.timesteps)

                # output = output[:, :, args.timesteps:2*args.timesteps, :]
                target = data[:, :, 1:, :]
            else:
                data_plot = data[:, :, args.timesteps:args.timesteps + 21, :].contiguous()
                output = decoder(data_plot, edges, rel_rec, rel_send, 20)
                target = data_plot[:, :, 1:, :]

            predicted_outputs.append(output.data.numpy())
            actual_outputs.append(target.data.numpy())
            logits_list.append(logits.data.numpy())
            edges_list.append(edges.data.numpy())

            mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
            tot_mse += mse.data.cpu().numpy()
            counter += 1

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))
    print('MSE: {}'.format(mse_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              file=log)
        print('MSE: {}'.format(mse_str), file=log)
        log.flush()

        with open(os.path.join(save_folder,"predicted_outputs_static.txt"), "wb") as f:
            pickle.dump(predicted_outputs,f)

        with open(os.path.join(save_folder,"actual_outputs_static.txt"), "wb") as f:
            pickle.dump(actual_outputs,f)
            
        with open(os.path.join(save_folder,"logits_static.txt"), "wb") as f:
            pickle.dump(logits_list,f)

        with open(os.path.join(save_folder,"edges_static.txt"), "wb") as f:
            pickle.dump(edges_list,f)

test()
if log is not None:
    print(save_folder)
    log.close()
