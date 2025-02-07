import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # experiment arguments
    parser.add_argument('--exp_name', default='TFQ', type=str)
    parser.add_argument('--exp_No', default=None, type=int)
    parser.add_argument('--stage', default='eval', type=str)
    parser.add_argument('--encode_type', default='PQ', type=str)
    # path
    parser.add_argument('--data_dir', default='../data/SIFT_1M', type=str)
    parser.add_argument('--dataset', default='SIFT1M', type=str)
    parser.add_argument('--weights_dir', default='./weights', type=str)
    parser.add_argument('--results_dir', default='./results', type=str)
    parser.add_argument('--cb_dir', default='./codebooks', type=str)
    parser.add_argument('--code_dir', default='./codes', type=str)
    # model parameters
    parser.add_argument('--model', default='TFDec', type=str)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--ffn_dim', default=512, type=int)
    parser.add_argument('--decoder_layers', default=6, type=int)
    parser.add_argument('--source_dim', default=128, type=int)
    parser.add_argument('--codebooks', default=8, type=int)
    parser.add_argument('--cb_bits', default=8, type=int)
    parser.add_argument('--topk', default=1, type=int)
    parser.add_argument('--sum_topk', default=False, action='store_true')
    # dataset parameters
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--workers', default=0, type=int)
    # device name
    parser.add_argument('--device', default='cuda:0', type=str)

    # training & evaluate parameters
    parser.add_argument('--loss_fn', default='MSE', type=str)
    parser.add_argument('--optimizer', default='AdamW', type=str)
    parser.add_argument('--kmeans_iter', default=100, type=int)
    parser.add_argument('--kmeans_redo', default=3, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--early_stop', default=None, type=int)
    parser.add_argument('--checkpoint', default=50, type=int) #sharing parameter
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--max_norm', default=0.1, type=float)

    # other
    parser.add_argument('--plotter_x_interval', default=20, type=int)

    return parser
