import argparse



def get_args(parser):

	parser.add_argument('--controller-max-episodes', type=int, default=2000, 
						help='num of training episodes for controller')
	parser.add_argument('--child-epochs-per-episode', type=int, default=1, 
						help='num of child training epochs per episode')
	parser.add_argument('--controller-iters-per-episode', type=int, default=50, 
						help='num of controller training episodes per episode')
	parser.add_argument('--controller-step-freq', type=int, default=10,
						help='frequency of controller param update; \
								i.e. batch size for controller')

	parser.add_argument('--controller-lr', type=float, default=0.0035)
	parser.add_argument('--controller-tanh-constant', type=float, default=0.44)
	parser.add_argument('--controller-lstm-size', type=int, default=100)
	parser.add_argument('--controller-lstm-layer-num', type=int, default=1)
	parser.add_argument('--controller-temperature', type=float, default=5.0)

	parser.add_argument('--batch-size', type=int, default=128, help='batch size')
	parser.add_argument('--child-branch-num', type=int, default=5, 
						help='per cell candidate operation num')
	parser.add_argument('--child-unit-num', type=int, default=5,
						help='per cell operation unit pair num')
	parser.add_argument('--child-lr-max', type=float, default=0.05)
	parser.add_argument('--child-lr-min', type=float, default=0.0005)
	parser.add_argument('--child-momentum', type=float, default=0.9, 
						help='momentum value for child optimizer')
	parser.add_argument('--child-weight_decay', type=float, default=1e-4, 
						help='weight decay value for child optimizer')
	parser.add_argument('--child-T0', type=int, default=10)
	parser.add_argument('--child-T-mul', type=int, default=2)
	parser.add_argument('--child-num-layers', '--N', type=int, default=6)
	parser.add_argument('--child-out-filters', type=int, default=20)
	parser.add_argument('--child-use-aux-heads', type=bool, default=False)

	parser.add_argument('--entropy-weight', type=float, default=0.0001)
	parser.add_argument('--bl-decay', type=float, default=0.99)
	# parser.add_argument('--report-freq', type=float, default=10, help='report frequency')


	return parser