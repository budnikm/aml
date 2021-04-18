import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Asymmetric metric learning')
    
    # export directory, training and val datasets, test datasets
    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')
    parser.add_argument('--training-dataset', '-d', metavar='DATASET', default='retrieval-SfM-120k', choices=training_dataset_names,
                        help='training dataset: ' + 
                            ' | '.join(training_dataset_names) +
                            ' (default: retrieval-SfM-120k)')
    parser.add_argument('--no-val', dest='val', action='store_false',
                        help='do not run validation')
    parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='roxford5k,rparis6k',
                        help='comma separated list of test datasets: ' + 
                            ' | '.join(test_datasets_names) + 
                            ' (default: roxford5k,rparis6k)')
    parser.add_argument('--test-whiten', metavar='DATASET', default='', choices=test_whiten_names,
                        help='dataset used to learn whitening for testing: ' + 
                            ' | '.join(test_whiten_names) + 
                            ' (default: None)')
    parser.add_argument('--val-freq', default=1, type=int, metavar='N', 
                        help='run val evaluation every N epochs (default: 1)')
    parser.add_argument('--save-freq', default=1, type=int, metavar='N', 
                        help='save model every N epochs (default: 1)')
                        
    # network architecture and initialization options
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet101)')
    parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
                        help='pooling options: ' +
                            ' | '.join(pool_names) +
                            ' (default: gem)')
    parser.add_argument('--local-whitening', '-lw', dest='local_whitening', action='store_true',
                        help='train model with learnable local whitening (linear layer) before the pooling')
    parser.add_argument('--regional', '-r', dest='regional', action='store_true',
                        help='train model with regional pooling using fixed grid')
    parser.add_argument('--whitening', '-w', dest='whitening', action='store_true',
                        help='train model with learnable whitening (linear layer) after the pooling')
    parser.add_argument('--not-pretrained', dest='pretrained', action='store_false',
                        help='initialize model with random weights (default: pretrained on imagenet)')
    parser.add_argument('--loss', '-l', metavar='LOSS', default='contrastive',
                        choices=loss_names,
                        help='training loss options: ' +
                            ' | '.join(loss_names) +
                            ' (default: contrastive)')
    parser.add_argument('--mode', '-m', metavar='MODE', default='std',
                        choices=mode_names,
                        help='training mode options: ' +
                            ' | '.join(mode_names) +
                            ' (default: std)')
    parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
                        help='loss margin: (default: 0.7)')

    # train/val options specific for image retrieval learning
    parser.add_argument('--image-size', default=1024, type=int, metavar='N',
                        help='maximum size of longer image side used for training (default: 1024)')
    parser.add_argument('--neg-num', '-nn', default=5, type=int, metavar='N',
                        help='number of negative image per train/val tuple (default: 5)')
    parser.add_argument('--query-size', '-qs', default=2000, type=int, metavar='N',
                        help='number of queries randomly drawn per one train epoch (default: 2000)')
    parser.add_argument('--pool-size', '-ps', default=20000, type=int, metavar='N',
                        help='size of the pool for hard negative mining (default: 20000)')

    # standard train/val options
    parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                        help='gpu id used for training (default: 0)')
    parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--batch-size', '-b', default=5, type=int, metavar='N', 
                        help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)')
    parser.add_argument('--update-every', '-u', default=1, type=int, metavar='N',
                        help='update model weights every N batches, used to handle really large batches, ' + 
                            'batch_size effectively becomes update_every x batch_size (default: 1)')
    parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                        choices=optimizer_names,
                        help='optimizer options: ' +
                            ' | '.join(optimizer_names) +
                            ' (default: adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-6)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-6)')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                        help='name of the latest checkpoint (default: None)')
    parser.add_argument('--comment', '-c', default='', type=str, metavar='COMMENT',
                        help='additional experiment comment')
    parser.add_argument('--temp', default=0.1, type=float, metavar='TEMP',
                        help='temperature for the softmax loss function')
    parser.add_argument('--nexamples', default=1000, type=int, metavar='N', # Probably don't need !!! 
                        help='number of negative examples for AP or cross(default: 1000)')
    parser.add_argument('--teacher', '-t', metavar='TEACHER', default='vgg16',
                        choices=teacher_names,
                        help='training mode options: ' +
                            ' | '.join(teacher_names) +
                            ' (default: vgg16)')
    parser.add_argument('--sym', dest='sym', action='store_true',
                        help='symmetric training')

    parser.add_argument('--pos-num', '-pn', default=3, type=int, metavar='N',
                        help='number of positive images per train/val tuple (default: 5)')

    return parser
    
def parse_commandline_args():
    return create_parser().parse_args()