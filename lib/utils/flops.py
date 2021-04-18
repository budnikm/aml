import argparse

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

from lib.networks.imageretrievalnet import init_network

parser = argparse.ArgumentParser(description='PyTorch CNN Flops calculation')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help="pretrained network or network path (destination where network is saved)")


def main():
	args = parser.parse_args()

	state = torch.load(args.network_path)

	net_params = {}
	net_params['architecture'] = state['meta']['architecture']
	net_params['pooling'] = state['meta']['pooling']
	net_params['local_whitening'] = state['meta'].get('local_whitening', False)
	net_params['regional'] = state['meta'].get('regional', False)
	net_params['whitening'] = state['meta'].get('whitening', False)
	net_params['mean'] = state['meta']['mean']
	net_params['std'] = state['meta']['std']
	net_params['pretrained'] = False
	net_params['teacher'] = 'resnet101'
	
	net = init_network(net_params)
	net.cuda()
	flops, params = get_model_complexity_info(net, (3, 362, 362), as_strings=True, print_per_layer_stat=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
	main()