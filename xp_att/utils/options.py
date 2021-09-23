
import argparse

def parse_args():
    p = argparse.ArgumentParser(description='SLR')
    p.add_argument('-t', '--task', type=str, default='train')
    p.add_argument('-g', '--gpu', type=int, default=0)
    p.add_argument('-sd', '--seed', type=int, default=8)

    # data
    p.add_argument('-dw', '--data_worker', type=int, default=32)
    p.add_argument('-fd', '--feature_dim', type=int, default=512)
    p.add_argument('-corp_dir', '--corpus_dir', type=str, default='Data/slr-phoenix14')
    p.add_argument('-voc_fl', '--vocab_file', type=str, default='Data/slr-phoenix14/newtrainingClasses.txt')
    p.add_argument('-corp_tr', '--corpus_train', type=str, default='Data/slr-phoenix14/train.corpus.csv')
    p.add_argument('-corp_te', '--corpus_test', type=str, default='Data/slr-phoenix14/test.corpus.csv')
    p.add_argument('-corp_de', '--corpus_dev', type=str, default='Data/slr-phoenix14/dev.corpus.csv')
    p.add_argument('-vp', '--video_path', type=str, default='../c3d_res_phoenix_body_iter5_120k')

    p.add_argument('--stage_epoch', type=int, default=10)


    # optimizer
    p.add_argument('-op', '--optimizer', type=str, default='adam')
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    p.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    p.add_argument('-mt', '--momentum', type=float, default=0.9)
    p.add_argument('-nepoch', '--max_epoch', type=int, default=1000)
    p.add_argument('-mupdates', '--max_updates', type=int, default=1e7)
    p.add_argument('-us', '--update_step', type=int, default=1)
    p.add_argument('-upm', '--update_param', type=str, default='all')

    p.add_argument('-bnmt', '--bn_momentum', type=float, default=0.1)

    # self-attention
    # mymodule parameters
    p.add_argument("--max_relative_positions", default=8, type=int, help="max relative distance")
    p.add_argument("--relative_attention", default=True, type=bool, help="whether using relative attention")
    p.add_argument("--pos_att_type", default="c2p|p2c", type=str, help="position attention type")
    p.add_argument("--window_size", default=16, type=int, help="window size")
    p.add_argument("--local_num_layers", default=0, type=int, help="local_num_layers")
    p.add_argument("--use_relative", default=True, type=bool, help="use_relative")
    p.add_argument("--is_adaptive", default=True, type=bool, help="is_adptive")

    # model parameters
    p.add_argument("--embedding_dim", default=512, type=int, help="embedding dim")
    p.add_argument("--input_size", default=1024, type=int, help="input feature dim")
    p.add_argument("--hidden_size", default=512, type=int, help="hidden dim")
    p.add_argument("--ff_size", default=2048, type=int, help="dim of feedforward")
    p.add_argument("--num_heads", default=8, type=int, help="number of heads")
    p.add_argument("--num_layers", default=6, type=int, help="number of layers")
    p.add_argument("--dropout", default=0.1, type=float, help="attention dropout")
    p.add_argument("--emb_dropout", default=0.1, type=float, help="embedding dropout")

    p.add_argument("--freeze", default=False, type=bool, help="weather freeze the parameters")
    p.add_argument("--norm_type", default="batch", type=str, help="normalization for spatial/word embedding")
    p.add_argument("--activation_type", default="softsign", type=str, help="activation function for spatial/word embedding")
    p.add_argument("--scale", default=False, type=bool, help="weather scale the embedding feature")
    p.add_argument("--scale_factor", default=None, type=float, help="scale factor for the embedding feature")
    p.add_argument("--fp16", default=False, type=bool, help="fp16")


    # train
    p.add_argument('-rl', '--reset_lr', type=bool, default=False)
    p.add_argument('-db', '--DEBUG', type=bool, default=False)
    p.add_argument('-lg_d', '--log_dir', type=str, default='./log/debug')
    p.add_argument('-bs', '--batch_size', type=int, default=20)
    p.add_argument('-ckpt', '--check_point', type=str, default='')
    p.add_argument('-pt', '--pretrain', type=str, default='')
    p.add_argument('-ps', '--print_step', type=int, default=20)
    p.add_argument('-siu', '--save_interval_updates', type=int, default=100)
    p.add_argument('-frc', '--freeze_cnn', type=bool, default=False)
    p.add_argument('-olbc', '--only_load_backbone', type=bool, default=False)
    p.add_argument('-clip', '--clip', type=float, default=5.0)

    # test (decoding)
    p.add_argument('-bwd', '--beam_width', type=int, default=5)
    p.add_argument('-vbs', '--valid_batch_size', type=int, default=1)
    p.add_argument('-evalset', '--eval_set', type=str, default='test', choices=['test', 'dev'])

    parameter = p.parse_args()
    return parameter


if __name__ == "__main__":
    opts = parse_args()
    print(opts)
